"""v0.4: nominations at Moffat: implied calorific value, daily controls, day-ahead nomination forecasts, assumed imbalance charge."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nbpie import nominations as N  # noqa: E402

RES = ROOT / "results"
ISEM = Path.home() / "ie-isem-market" / "data" / "raw"
mof = pd.read_csv(ROOT / "data" / "national_gas_moffat.csv", index_col=0, parse_dates=True).loc["2021-11-22":]
out = {}

# 1. controls
c = N.controls(mof)
out["days"] = int(len(c))
out["flags"] = {k: int(c[k].sum()) for k in ["missing_day", "missing_value", "nonpositive", "cv_outside", "nom_dev", "any_flag"]}
cv = c["cv"][~c["cv_outside"]].dropna()
out["implied_cv_kwh_m3"] = {"median": round(float(cv.median()), 3), "p05": round(float(cv.quantile(0.05)), 3), "p95": round(float(cv.quantile(0.95)), 3)}
c[c.any_flag].round(4).to_csv(RES / "nomination_flagged_days.csv")
ok = c[~c.any_flag]
out["nomination_vs_flow"] = {"days": int(len(ok)), "mean_abs_dev_pct": round(float(ok.nom_over_est.abs().mean() * 100), 2),
                             "p95_abs_dev_pct": round(float(ok.nom_over_est.abs().quantile(0.95) * 100), 2)}

# 2. day-ahead forecasts of the prevailing nomination (GWh per gas day)
bad = c["cv_outside"].reindex(mof.index).fillna(False)
y = (mof["moffat_nomination_exit_kwh"] / 1e6).where(~bad).asfreq("D")


def daily(df):
    df = df.copy()
    df["ts"] = pd.to_datetime(df.ts_utc, utc=True).dt.tz_convert(None)
    if "series" in df.columns:
        df = df.groupby(["ts", "series"]).value.sum().groupby("ts").sum().reset_index(name="value")
    return df.groupby(df.ts.dt.normalize()).value.mean()


load_fc = None
res_fc = daily(pd.read_parquet(ISEM / "res_fc_2023-01-01_2026-10-01.parquet"))
sap_raw = pd.read_excel(ROOT / "data" / "ons_sap.xlsx", sheet_name="1.Daily SAP Gas", header=None)
sap_raw = sap_raw[pd.to_datetime(sap_raw[0], errors="coerce").notna()]
sap_p = pd.Series(pd.to_numeric(sap_raw[1], errors="coerce").values, index=pd.to_datetime(sap_raw[0])).dropna()
fx = pd.read_csv(ROOT / "data" / "ecb_eurgbp.csv", index_col=0, parse_dates=True).iloc[:, 0]
sap = sap_p * 10.0 * fx.reindex(sap_p.index).ffill().bfill()            # EUR/MWh
X = N.features(y, load_fc, res_fc, sap)
START = "2024-01-01"
BASE = ["lag2", "lag3", "lag7", "lag14", "mean7_l2", "s1", "c1"]
FULL = BASE + ["res_fc", "sap_l2"]


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=0)


dow = pd.get_dummies(X["dow"], prefix="d", dtype=float)
specs = {
    "lag7 (same weekday last week)": None,
    "lag2 (two days ago)": None,
    "ridge: nomination lags and calendar": (pd.concat([X[BASE], dow], axis=1), ridge),
    "ridge: plus day-ahead renewable forecast and SAP": (pd.concat([X[FULL], dow], axis=1), ridge),
    "gradient boosting: same inputs": (pd.concat([X[FULL], X[["dow"]]], axis=1), gbm),
}
preds = {}
for name, s in specs.items():
    if s is None:
        preds[name] = X["lag7"] if name.startswith("lag7") else X["lag2"]
    else:
        preds[name] = N.walk_forward(y, s[0], s[1], START)
P = pd.DataFrame(preds)
mask = (y.index >= START) & y.notna() & P.notna().all(axis=1)
yt = y[mask]
rows = []
for name in P.columns:
    e = P.loc[mask, name] - yt
    rows.append({"model": name, "days": int(mask.sum()), "mae_gwh": round(float(e.abs().mean()), 2),
                 "mape_pct": round(float((e.abs() / yt).mean() * 100), 2), "bias_gwh": round(float(e.mean()), 2),
                 "p95_abs_error_gwh": round(float(e.abs().quantile(0.95)), 1),
                 "assumed_charge_eur_m_per_yr": round(float(N.imbalance_cost(e, sap).sum() / 1e6 / (mask.sum() / 365.25)), 2)})
tab = pd.DataFrame(rows)
tab.to_csv(RES / "nomination_forecasts.csv", index=False)
out["forecasts"] = rows
out["forecast_period"] = [str(yt.index.min().date()), str(yt.index.max().date())]
best = tab.sort_values("mae_gwh").iloc[0].model
e = (P.loc[mask, best] - yt).abs()
out["best_model"] = best
season = np.where(yt.index.month.isin([12, 1, 2]), "winter", np.where(yt.index.month.isin([6, 7, 8]), "summer", "spring_autumn"))
out["best_mae_by_season_gwh"] = {k: round(float(v), 2) for k, v in e.groupby(season).mean().items()}
out["share_of_days_error_above_5pct_best"] = round(float(((e / yt) > 0.05).mean()), 3)
(RES / "nominations_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

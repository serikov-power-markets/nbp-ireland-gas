"""Gas chain from NBP to Ireland: SAP, Moffat flows, link to the I-SEM, capacity booking exercise, reporting schema check."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nbpie import booking, remit  # noqa: E402

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
ISEM = Path.home() / "ie-isem-market"
GWH_PER_MSCM = 11.0     # ASSUMED calorific conversion (about 39.6 MJ/m3); the published Moffat calorific value was not used
EFF = 0.5               # ASSUMED CCGT efficiency for gas burn and spark spread
FX_FALLBACK = 1.17      # EUR per GBP, used only if the ECB file is missing (assumed)

mof = pd.read_csv(ROOT / "data" / "national_gas_moffat.csv", index_col=0, parse_dates=True)
flow = (mof["moffat_physical_flow_mscm"] * GWH_PER_MSCM).dropna()
flow = flow[flow > 0]
sap_raw = pd.read_excel(ROOT / "data" / "ons_sap.xlsx", sheet_name="1.Daily SAP Gas", header=None)
sap = sap_raw[pd.to_datetime(sap_raw[0], errors="coerce").notna()].copy()
sap.index = pd.to_datetime(sap[0])
sap = pd.to_numeric(sap[1], errors="coerce").dropna().rename("sap_pkwh")          # pence per kWh
fx_path = ROOT / "data" / "ecb_eurgbp.csv"
if fx_path.exists():
    fx = pd.read_csv(fx_path, index_col=0, parse_dates=True).iloc[:, 0].rename("eur_per_gbp")
else:
    fx = pd.Series(FX_FALLBACK, index=sap.index, name="eur_per_gbp")
d = pd.concat([flow.rename("flow_gwh"), sap, fx], axis=1).dropna(subset=["flow_gwh"])
d["eur_per_gbp"] = d["eur_per_gbp"].ffill().bfill()
d["sap_eur_mwh"] = d["sap_pkwh"] * 10.0 * d["eur_per_gbp"]                         # p/kWh x 10 = GBP/MWh
out = {"period": [str(flow.index.min().date()), str(flow.index.max().date())], "days": int(len(flow)), "fx": "ECB daily" if fx_path.exists() else "constant 1.17 (assumed)"}

# A. Moffat flow profile
ann = flow.groupby(flow.index.year).sum() / 1000.0
seas = flow.groupby(flow.index.month).mean()
out["annual_twh"] = {int(k): round(float(v), 1) for k, v in ann.items()}
out["peak_day_gwh"] = round(float(flow.max()), 0)
out["peak_day"] = str(flow.idxmax().date())
out["mean_gwh_d"] = round(float(flow.mean()), 1)
out["winter_over_summer"] = round(float(seas.loc[[12, 1, 2]].mean() / seas.loc[[6, 7, 8]].mean()), 2)
out["p95_over_mean"] = round(float(flow.quantile(0.95) / flow.mean()), 2)
seas.round(1).to_csv(RES / "moffat_monthly_mean_gwh.csv", header=["gwh_per_day"])

# B. link to the I-SEM: gas-fired generation and price (2023 onwards)
g = pd.read_parquet(ISEM / "data" / "raw" / "gen_2023-01-01_2026-10-01.parquet")
g = g[g.series == "gas"].copy()
g["ts"] = pd.to_datetime(g.ts_utc, utc=True)
g["mwh"] = g.value * g.step_min / 60.0
gas_pow = g.groupby(g.ts.dt.tz_convert(None).dt.normalize()).mwh.sum() / 1000.0 / EFF    # GWh of gas burned at EFF
p = pd.read_parquet(ISEM / "data" / "raw" / "price_2023-01-01_2026-10-01.parquet")
p["ts"] = pd.to_datetime(p.ts_utc, utc=True)
price = p.groupby(p.ts.dt.tz_convert(None).dt.normalize()).value.mean().rename("isem_price")
e = pd.concat([flow.rename("flow"), gas_pow.rename("pow_gas"), price, d["sap_eur_mwh"]], axis=1).dropna()
e = e[(e.pow_gas > 0)]
X = pd.get_dummies(e.index.month, prefix="m", drop_first=True, dtype=float)
X.index = e.index
X = sm.add_constant(pd.concat([e[["pow_gas"]], X], axis=1))
m1 = sm.OLS(e["flow"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 7})
m0 = sm.OLS(e["flow"], sm.add_constant(X.drop(columns="pow_gas"))).fit()
out["link_isem"] = {"days": int(len(e)), "coef_gwh_per_gwh_power_gas": round(float(m1.params["pow_gas"]), 2), "t": round(float(m1.tvalues["pow_gas"]), 1),
                    "r2_with": round(float(m1.rsquared), 3), "r2_month_only": round(float(m0.rsquared), 3),
                    "mean_power_gas_gwh_d": round(float(e.pow_gas.mean()), 1), "mean_moffat_gwh_d": round(float(e.flow.mean()), 1),
                    "corr_flow_isem_price": round(float(e.flow.corr(e.isem_price)), 2)}
# spark spread: fuel and carbon cost of gas generation against the I-SEM day-ahead price
eua = pd.read_pickle(ISEM / "results" / "eua.pkl")
eua = eua.set_index(pd.to_datetime(eua["date"]))["price"].reindex(e.index).ffill().bfill()
e["spark_fuel"] = e.isem_price - e.sap_eur_mwh / EFF
e["spark_clean"] = e.spark_fuel - 0.202 * 3.6 / 3.6 / EFF * eua      # 0.202 t CO2 per MWh gas (assumed factor)
sp = e[["spark_fuel", "spark_clean"]].groupby(e.index.year).mean().round(1)
sp.to_csv(RES / "spark_spread_by_year.csv")
out["spark_by_year"] = sp.to_dict()
out["sap_eur_mwh_by_year"] = e.sap_eur_mwh.groupby(e.index.year).mean().round(1).to_dict()
e.round(3).to_csv(RES / "daily_link.csv")

# C. capacity booking exercise: train on the previous gas year, test on the next
fc = booking.naive_forecast(flow)
gy = pd.Series(booking.gas_year(flow.index), index=flow.index)
rows = []
for y in sorted(gy.unique())[1:]:
    tr, te = gy[gy == y - 1].index, gy[gy == y].index
    if len(tr) < 330 or len(te) < 330:
        continue
    k_tr, _ = booking.best_base(flow[tr], fc[tr])
    k_peak, k_p95 = float(flow[tr].max()), float(flow[tr].quantile(0.95))
    k_opt, c_hind = booking.best_base(flow[te], fc[te])
    res = {"gas_year": int(y), "base_trained_gwh": round(k_tr, 1), "base_hindsight_gwh": round(k_opt, 1), "base_peak_gwh": round(k_peak, 1)}
    for name, k in (("trained", k_tr), ("peak_last_year", k_peak), ("p95_last_year", k_p95)):
        res[f"cost_{name}"] = round(booking.cost(flow[te], k, fc[te]), 0)
    res["cost_hindsight"] = round(c_hind, 0)
    res["cost_trained_perfect_forecast"] = round(booking.cost(flow[te], k_tr, flow[te]), 0)
    res["regret_trained_pct"] = round(100 * (res["cost_trained"] / c_hind - 1), 1)
    res["regret_peak_pct"] = round(100 * (res["cost_peak_last_year"] / c_hind - 1), 1)
    res["regret_p95_pct"] = round(100 * (res["cost_p95_last_year"] / c_hind - 1), 1)
    rows.append(res)
bk = pd.DataFrame(rows)
bk.to_csv(RES / "booking_by_gas_year.csv", index=False)
out["booking_mean_regret_pct"] = {c: round(float(bk[c].mean()), 1) for c in ("regret_trained_pct", "regret_peak_pct", "regret_p95_pct")}
out["booking_years"] = int(len(bk))

# D. reporting schema check
df, inj = remit.synthetic_trades(errors=True)
v = remit.validate(df, today="2027-01-01")
v.to_csv(RES / "validation_findings.csv", index=False)
out["remit"] = {"trades": int(len(df)), "injected_defects": len(inj), "findings": int(len(v)),
                "all_injected_found": all(((v.row == r) & (v.field == f)).any() for r, f in inj.items()),
                "clean_file_findings": int(len(remit.validate(remit.synthetic_trades(errors=False)[0], today="2027-01-01")))}
df.to_csv(RES / "synthetic_trades.csv", index=False)
(RES / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
print(json.dumps(out, indent=2, default=float))

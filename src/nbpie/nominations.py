"""Day-ahead gas nominations at an exit point: implied calorific value, forecast models and process controls.

What is real: the prevailing exit nomination at Moffat and the physical flow (National Gas Transmission REST data).
What is assumed: the imbalance charge (a share of the gas price per MWh of nomination error) and that the day-ahead information set is
lags of at least two days plus the day-ahead load and renewable forecasts for the gas day. The data hold the prevailing (final)
nomination, not the first nomination of the gas day, so the error here is the error of a forecast that would be placed the day
before, measured against what the shippers finally nominated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_cv(nom_kwh: pd.Series, flow_mscm: pd.Series) -> pd.Series:
    """kWh per cubic metre implied by the two series (nomination energy divided by physical volume)."""
    return nom_kwh / (flow_mscm * 1e6)


def controls(df: pd.DataFrame, cv_lo=10.0, cv_hi=12.0, tol=0.10) -> pd.DataFrame:
    """Daily control checks, one row per gas day with boolean flags.

    missing_day: gas day absent from the index; missing_value: nomination or flow absent; nonpositive: zero or negative value;
    cv_outside: implied calorific value outside [cv_lo, cv_hi] kWh/m3; nom_dev: nomination deviates from flow x rolling median CV
    by more than `tol`.
    """
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    d = df.reindex(full)
    out = pd.DataFrame(index=d.index)
    out["missing_day"] = ~d.index.isin(df.index)
    nom, fl = d["moffat_nomination_exit_kwh"], d["moffat_physical_flow_mscm"]
    out["missing_value"] = nom.isna() | fl.isna()
    out["nonpositive"] = (nom <= 0) | (fl <= 0)
    cv = implied_cv(nom, fl)
    out["cv"] = cv
    out["cv_outside"] = (cv < cv_lo) | (cv > cv_hi)
    med = cv.where(~out["cv_outside"]).rolling(30, min_periods=10, center=True).median().bfill().ffill()
    est = fl * 1e6 * med
    out["nom_over_est"] = nom / est - 1
    out["nom_dev"] = out["nom_over_est"].abs() > tol
    out["any_flag"] = out[["missing_day", "missing_value", "nonpositive", "cv_outside", "nom_dev"]].any(axis=1)
    return out


def features(nom_gwh: pd.Series, load_fc, res_fc, sap) -> pd.DataFrame:
    """Information available the day before the gas day: nominations lagged by at least 2 days, calendar, day-ahead forecasts."""
    idx = nom_gwh.index
    X = pd.DataFrame(index=idx)
    for k in (2, 3, 7, 14):
        X[f"lag{k}"] = nom_gwh.shift(k)
    X["mean7_l2"] = nom_gwh.shift(2).rolling(7).mean()
    X["dow"] = idx.dayofweek
    doy = idx.dayofyear
    X["s1"], X["c1"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    if load_fc is not None:
        X["load_fc"] = load_fc.reindex(idx)
    if res_fc is not None:
        X["res_fc"] = res_fc.reindex(idx)
        if load_fc is not None:
            X["resid_fc"] = X["load_fc"] - X["res_fc"]
    if sap is not None:
        X["sap_l2"] = sap.reindex(idx).shift(2)
    return X


def walk_forward(y: pd.Series, X: pd.DataFrame, make_model, start: str) -> pd.Series:
    """Expanding-window forecast refitted at the start of each month; rows with missing features are skipped."""
    pred = pd.Series(np.nan, index=y.index)
    starts = pd.date_range(start, y.index.max(), freq="MS")
    okx = X.notna().all(axis=1).values
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else y.index.max() + pd.Timedelta(days=1)
        trm = (X.index < s) & okx & y.notna().values
        te = (X.index >= s) & (X.index < e) & okx
        if trm.sum() < 200 or te.sum() == 0:
            continue
        m = make_model()
        m.fit(X[trm], y[trm])
        pred[te] = m.predict(X[te])
    return pred


def imbalance_cost(err_gwh: pd.Series, sap_eur_mwh: pd.Series, share=0.03) -> pd.Series:
    """ASSUMED charge: `share` of the gas price on every MWh of absolute nomination error (EUR)."""
    return err_gwh.abs() * 1000.0 * sap_eur_mwh.reindex(err_gwh.index).ffill() * share

"""A field schema for REMIT-style trade reporting and an EMIR-style derivative report, with a validation function.

NOT VERIFIED against the current ACER REMIT user manual or the EMIR technical standards: the field list is a simplified working
schema written for this exercise. A real submission must follow the current official tables.
"""
import re

import numpy as np
import pandas as pd

ISO_CCY = {"EUR", "GBP", "USD"}
CODE_TYPES = {"ACE", "LEI", "BIC", "EIC", "GLN"}
UNITS = {"MWh/d", "MW", "therm/d"}
REQUIRED = ["reporting_party_id", "code_type", "lei", "uti", "trade_date", "buy_sell", "commodity", "contract_type", "delivery_point",
            "delivery_start", "delivery_end", "price", "price_ccy", "quantity", "quantity_unit", "counterparty_lei"]


def validate(df: pd.DataFrame, today=None) -> pd.DataFrame:
    """One row per violation: row index, field, rule."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    out = []

    def add(i, f, rule):
        out.append({"row": int(i), "field": f, "rule": rule})

    for f in REQUIRED:
        if f not in df.columns:
            add(-1, f, "column missing")
    for i, r in df.iterrows():
        for f in REQUIRED:
            if f in df.columns and (pd.isna(r[f]) or str(r[f]).strip() == ""):
                add(i, f, "required value missing")
        if pd.notna(r.get("code_type")) and r["code_type"] not in CODE_TYPES:
            add(i, "code_type", "unknown code type")
        for f in ("lei", "counterparty_lei"):
            if pd.notna(r.get(f)) and not re.fullmatch(r"[A-Z0-9]{20}", str(r[f])):
                add(i, f, "LEI must be 20 characters A-Z0-9")
        if pd.notna(r.get("lei")) and r.get("lei") == r.get("counterparty_lei"):
            add(i, "counterparty_lei", "counterparty equals reporting party")
        if pd.notna(r.get("uti")) and not re.fullmatch(r"[A-Z0-9]{1,52}", str(r["uti"])):
            add(i, "uti", "UTI must be 1-52 characters A-Z0-9")
        if pd.notna(r.get("buy_sell")) and r["buy_sell"] not in ("B", "S"):
            add(i, "buy_sell", "must be B or S")
        if pd.notna(r.get("commodity")) and r["commodity"] != "NG":
            add(i, "commodity", "must be NG")
        if pd.notna(r.get("contract_type")) and r["contract_type"] not in {"FW", "FU", "SW", "OP"}:
            add(i, "contract_type", "unknown contract type")
        if pd.notna(r.get("price_ccy")) and r["price_ccy"] not in ISO_CCY:
            add(i, "price_ccy", "currency not in list")
        for f in ("price", "quantity"):
            if pd.notna(r.get(f)):
                try:
                    if float(r[f]) <= 0:
                        add(i, f, "must be positive")
                except (TypeError, ValueError):
                    add(i, f, "not a number")
        if pd.notna(r.get("quantity_unit")) and r["quantity_unit"] not in UNITS:
            add(i, "quantity_unit", "unit not allowed")
        try:
            td, ds, de = pd.Timestamp(r["trade_date"]), pd.Timestamp(r["delivery_start"]), pd.Timestamp(r["delivery_end"])
            if td > today:
                add(i, "trade_date", "in the future")
            if ds < td:
                add(i, "delivery_start", "before trade date")
            if de <= ds:
                add(i, "delivery_end", "not after delivery start")
        except (KeyError, ValueError, TypeError):
            pass
    if "uti" in df:
        dup = df[df.duplicated("uti", keep=False) & df["uti"].notna()]
        for i in dup.index:
            add(i, "uti", "duplicate UTI")
    return pd.DataFrame(out, columns=["row", "field", "rule"])


def synthetic_trades(n=40, seed=3, errors=True):
    """A synthetic trade file (no real counterparties). With errors=True known defects are injected; returns (df, {row: field})."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-10-01")
    rows = []
    for k in range(n):
        td = base + pd.Timedelta(days=int(rng.integers(0, 40)))
        ds = td + pd.Timedelta(days=int(rng.integers(1, 90)))
        rows.append({"reporting_party_id": "TEST-PARTY-1", "code_type": "LEI", "lei": "SYNTHETICLEI0000001A",
                     "uti": f"UTI{k:08d}TEST", "trade_date": td.date().isoformat(), "buy_sell": str(rng.choice(["B", "S"])), "commodity": "NG",
                     "contract_type": str(rng.choice(["FW", "FU", "SW"])), "delivery_point": str(rng.choice(["NBP", "TTF", "SEP"])),
                     "delivery_start": ds.date().isoformat(), "delivery_end": (ds + pd.Timedelta(days=int(rng.integers(1, 30)))).date().isoformat(),
                     "price": round(float(rng.uniform(25, 90)), 2), "price_ccy": str(rng.choice(["EUR", "GBP"])),
                     "quantity": int(rng.integers(5, 400)) * 24, "quantity_unit": "MWh/d", "counterparty_lei": f"SYNTHETICCPTY{k % 7:07d}"})
    df = pd.DataFrame(rows)
    injected = {}
    if errors:
        df.loc[3, "price"] = -5; injected[3] = "price"
        df.loc[7, "lei"] = "SHORT"; injected[7] = "lei"
        df.loc[11, "delivery_end"] = df.loc[11, "delivery_start"]; injected[11] = "delivery_end"
        df.loc[15, "uti"] = df.loc[14, "uti"]; injected[15] = "uti"
        df.loc[20, "counterparty_lei"] = df.loc[20, "lei"]; injected[20] = "counterparty_lei"
        df.loc[25, "buy_sell"] = "X"; injected[25] = "buy_sell"
        df.loc[30, "price_ccy"] = "GBX"; injected[30] = "price_ccy"
    return df, injected

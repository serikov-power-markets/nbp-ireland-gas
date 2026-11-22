"""Validator for REMIT standard-contract reports built on the OFFICIAL field list of Table 1 of Commission Implementing Regulation (EU)
No 1348/2014 (fields 1 to 58, as published in the Official Journal L 363, 18.12.2014; text read from the ACER copy of the
regulation, data/raw/remit/Implementing_Regulation.pdf, 23 Nov 2026).

Only rules that the regulation text itself states are implemented: the code types of fields 2, 5, 7 and 9, voice-brokered Y or blank, the
action types of field 58, option style and type, settlement method and load type values, the termination date left blank when it equals
the delivery end date, the counterparty being a different party, and the reporting deadline of Article 7(1) (no later than the working
day after the conclusion). Formats, code lists and mandatory flags of the ACER Transaction Reporting User Manual and the XSD schemas
were NOT read: a real submission must follow them. The 2026 amendment of the regulation (listed on the ACER documents page) was not read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# field number -> identifier (Table 1)
FIELDS = {
    1: "ID of the market participant or counterparty", 2: "Type of code used in field 1",
    3: "ID of the trader and/or of the market participant or counterparty as identified by the organised market place",
    4: "ID of the other market participant or counterparty", 5: "Type of code used in field 4", 6: "Reporting entity ID", 7: "Type of code used in field 6",
    8: "Beneficiary ID", 9: "Type of code used in field 8", 10: "Trading capacity of the market participant or counterparty in field 1", 11: "Buy/sell indicator",
    12: "Initiator/Aggressor", 13: "Order ID", 14: "Order type", 15: "Order condition", 16: "Order status", 17: "Minimum execution volume", 18: "Price limit",
    19: "Undisclosed volume", 20: "Order duration", 21: "Contract ID", 22: "Contract name", 23: "Contract type", 24: "Energy commodity",
    25: "Fixing index or reference price", 26: "Settlement method", 27: "Organised market place ID/OTC", 28: "Contract trading hours",
    29: "Last trading date and time", 30: "Transaction timestamp", 31: "Unique transaction ID", 32: "Linked transaction ID", 33: "Linked order ID",
    34: "Voice-brokered", 35: "Price", 36: "Index value", 37: "Price currency", 38: "Notional amount", 39: "Notional currency", 40: "Quantity/Volume",
    41: "Total notional contract quantity", 42: "Quantity unit for field 40 and 41", 43: "Termination date", 44: "Option style", 45: "Option type",
    46: "Option exercise date", 47: "Option strike price", 48: "Delivery point or zone", 49: "Delivery start date", 50: "Delivery end date", 51: "Duration",
    52: "Load type", 53: "Days of the week", 54: "Load delivery Intervals", 55: "Delivery capacity", 56: "Quantity unit used in field 55",
    57: "Price/time interval quantity", 58: "Action type",
}
CODE_TYPES = {"ACER", "LEI", "BIC", "EIC", "GLN"}
ACTION = {"new", "modify", "error", "cancel"}
OPTION_STYLE = {"european", "asian", "bermudan", "american"}
OPTION_TYPE = {"call", "put", "other"}
SETTLEMENT = {"physical", "cash", "optional", "other"}
LOAD = {"base load", "peak load", "off-peak", "block of hours", "other"}


def _end_of_next_working_day(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts.normalize() + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def _val(r, key):
    v = r.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return None
    return v


def validate(df: pd.DataFrame, today=None) -> pd.DataFrame:
    """df has one column per field number, named f1 .. f58, plus report_time (when the record was submitted). One row per report."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    out = []

    def add(i, f, rule):
        out.append({"row": int(i), "field": f, "rule": rule})

    for i, r in df.iterrows():
        for n in (2, 5, 7, 9):
            v = _val(r, f"f{n}")
            if v is not None and v not in CODE_TYPES:
                add(i, f"f{n}", "code type must be ACER, LEI, BIC, EIC or GLN")
        v = _val(r, "f34")
        if v is not None and v != "Y":
            add(i, "f34", "voice-brokered is Y or blank")
        v = _val(r, "f58")
        if v is None or v not in ACTION:
            add(i, "f58", "action type must be new, modify, error or cancel")
        for n, allowed, nm in ((44, OPTION_STYLE, "option style"), (45, OPTION_TYPE, "option type"), (26, SETTLEMENT, "settlement method"), (52, LOAD, "load type")):
            v = _val(r, f"f{n}")
            if v is not None and str(v).lower() not in allowed:
                add(i, f"f{n}", f"{nm} not in the values named in the regulation")
        if _val(r, "f1") is not None and _val(r, "f1") == _val(r, "f4"):
            add(i, "f4", "other counterparty equals reporting counterparty")
        v = _val(r, "f11")
        if v is not None and str(v).lower() not in {"b", "s", "buy", "sell"}:
            add(i, "f11", "buy/sell indicator not recognised (the regulation names buy or sell; the code list is in the user manual)")
        for n in (35, 40):
            v = _val(r, f"f{n}")
            if v is not None:
                try:
                    if float(v) <= 0:
                        add(i, f"f{n}", "must be positive")
                except (TypeError, ValueError):
                    add(i, f"f{n}", "not a number")
        try:
            ds, de = pd.Timestamp(_val(r, "f49")), pd.Timestamp(_val(r, "f50"))
            if de < ds:
                add(i, "f50", "delivery end date before delivery start date")
            t = _val(r, "f43")
            if t is not None and pd.Timestamp(t) == de:
                add(i, "f43", "termination date must be blank when equal to the delivery end date")
        except (TypeError, ValueError):
            pass
        try:
            ts = pd.Timestamp(_val(r, "f30"))
            if ts > today:
                add(i, "f30", "transaction timestamp in the future")
            rt = _val(r, "report_time")
            if rt is not None and pd.Timestamp(rt) > _end_of_next_working_day(ts):
                add(i, "report_time", "reported later than the working day after conclusion (Article 7(1))")
        except (TypeError, ValueError):
            pass
    if "f31" in df:
        new = df[df.f58 == "new"]
        dup = new[new.duplicated("f31", keep=False) & new.f31.notna()]
        for i in dup.index:
            add(i, "f31", "duplicate unique transaction ID among new reports")
    return pd.DataFrame(out, columns=["row", "field", "rule"])


def synthetic(n=40, seed=5, errors=True):
    """Synthetic standard-contract reports (no real parties, codes are fake). Returns (df, {row: field}) with known defects when errors=True."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        ts = pd.Timestamp("2026-10-05 09:00") + pd.Timedelta(days=int(rng.integers(0, 40)), minutes=int(rng.integers(0, 600)))
        ds = (ts + pd.Timedelta(days=int(rng.integers(1, 90)))).normalize()
        row = {f"f{j}": None for j in FIELDS}
        row.update({"f1": "FAKEPARTY0000000001", "f2": "LEI", "f4": f"FAKECPTY{k % 7:011d}", "f5": "LEI", "f6": "FAKEPARTY0000000001", "f7": "LEI",
                    "f10": "P", "f11": str(rng.choice(["B", "S"])), "f21": f"CT{k}", "f23": "FW", "f24": "NG", "f26": "physical", "f27": "XBIL",
                    "f30": ts.isoformat(), "f31": f"UTI{k:08d}", "f35": round(float(rng.uniform(25, 90)), 2), "f37": "EUR", "f40": int(rng.integers(5, 400)) * 24,
                    "f42": "MWh/d", "f48": str(rng.choice(["NBP", "TTF"])), "f49": ds.date().isoformat(),
                    "f50": (ds + pd.Timedelta(days=int(rng.integers(1, 30)))).date().isoformat(),
                    "f52": "base load", "f58": "new", "report_time": (ts + pd.Timedelta(hours=3)).isoformat()})
        rows.append(row)
    df = pd.DataFrame(rows)
    inj = {}
    if errors:
        df.loc[3, "f35"] = -5; inj[3] = "f35"
        df.loc[7, "f2"] = "XYZ"; inj[7] = "f2"
        df.loc[11, "f50"] = (pd.Timestamp(df.loc[11, "f49"]) - pd.Timedelta(days=1)).date().isoformat(); inj[11] = "f50"
        df.loc[15, "f31"] = df.loc[14, "f31"]; inj[15] = "f31"
        df.loc[20, "f4"] = df.loc[20, "f1"]; inj[20] = "f4"
        df.loc[25, "f58"] = "add"; inj[25] = "f58"
        df.loc[30, "f34"] = "N"; inj[30] = "f34"
        df.loc[33, "report_time"] = (pd.Timestamp(df.loc[33, "f30"]) + pd.Timedelta(days=5)).isoformat(); inj[33] = "report_time"
        df.loc[36, "f43"] = df.loc[36, "f50"]; inj[36] = "f43"
    return df, inj

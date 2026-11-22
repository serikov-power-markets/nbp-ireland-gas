"""Moffat entry capacity booking with the PUBLISHED Gas Networks Ireland tariff structure of gas year 2025/26.

Source: GNI "Transmission Tariffs 2025/26" (CRU decision CRU/202543) and "Short Term Capacity Examples 2025/26", read 23 Nov 2026
(data/raw/gni_tariffs_2025_26/). Moffat Entry annual capacity: 436.790870 EUR per peak day MWh for the gas year. Short-term products
are the annual price times a multiplier per month (the daily and monthly multipliers below, copied from the examples sheet).

What is NOT from a source: (a) the overrun charge is set as a multiple of the daily price of that day (the Code of Operations was
not read; the multiple is varied 1.5 to 6); (b) the 2025/26 tariff is applied to every test year (earlier tariffs were not collected);
(c) the daily nomination-style forecast is the flow of the same weekday last week. Flow is the physical flow at Moffat, which is
not the booked capacity of any one shipper: the exercise books for the whole flow as if one shipper did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL = 436.790870          # EUR per peak day MWh per gas year, Moffat Entry (GNI 2025/26)
DAILY_MULT = {10: 0.00640418, 11: 0.00640418, 12: 0.01138520, 1: 0.01992410, 2: 0.02277040, 3: 0.01707780,
              4: 0.00640418, 5: 0.00048387, 6: 0.00048387, 7: 0.00048387, 8: 0.00048387, 9: 0.00048387}
MONTHLY_MULT = {10: 0.12808349, 11: 0.12808349, 12: 0.17077799, 1: 0.29886148, 2: 0.34155598, 3: 0.25616698,
                4: 0.12808349, 5: 0.00967742, 6: 0.00967742, 7: 0.00967742, 8: 0.00967742, 9: 0.00967742}


def daily_price(idx) -> np.ndarray:
    return np.array([ANNUAL * DAILY_MULT[m] for m in pd.DatetimeIndex(idx).month])


def monthly_price(idx) -> np.ndarray:
    """Price per MWh of capacity booked for the whole calendar month containing each day, spread per day of the month."""
    idx = pd.DatetimeIndex(idx)
    return np.array([ANNUAL * MONTHLY_MULT[m] / d for m, d in zip(idx.month, idx.days_in_month)])


def cost(flow, forecast, idx, base, monthly_level=None, overrun_mult=3.0):
    """Cost in EUR of a gas year. flow, forecast, base, monthly_level in MWh per day. Annual base K for the whole year, an optional
    monthly top-up level per calendar month (array of 12, index month-1), a daily top-up to the forecast on the rest, overrun on
    the remainder at overrun_mult times the daily price."""
    flow, forecast = np.asarray(flow, float), np.asarray(forecast, float)
    idx = pd.DatetimeIndex(idx)
    dp = daily_price(idx)
    held = np.full(len(flow), float(base))
    total = ANNUAL * float(base)
    if monthly_level is not None:
        lvl = np.maximum(np.asarray(monthly_level, float)[idx.month - 1], held)
        total += float(((lvl - held) * monthly_price(idx)).sum())
        held = lvl
    top = np.maximum(forecast - held, 0.0)
    total += float((top * dp).sum())
    held = held + top
    over = np.maximum(flow - held, 0.0)
    total += float((over * dp * overrun_mult).sum())
    return total


def best_base(flow, forecast, idx, grid=None, **kw):
    flow = np.asarray(flow, float)
    grid = np.linspace(0, flow.max() * 1.05, 150) if grid is None else grid
    c = [cost(flow, forecast, idx, k, **kw) for k in grid]
    i = int(np.argmin(c))
    return float(grid[i]), float(c[i])

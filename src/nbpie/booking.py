"""Capacity booking at an exit point: annual base booking plus daily top-up, with an overrun charge.

All prices are ASSUMED (relative units): the published tariffs of Gas Networks Ireland and National Gas were not read. Only the
ordering annual < daily < overrun is taken from how capacity products are priced in general.
"""
import numpy as np
import pandas as pd

C_ANNUAL = 1.0     # per unit of capacity per day, booked for the whole gas year (assumed)
C_DAILY = 2.5      # per unit booked on the day (assumed)
C_OVERRUN = 6.0    # per unit of flow above the capacity held (assumed)


def cost(flow, base, forecast, c_a=C_ANNUAL, c_d=C_DAILY, c_o=C_OVERRUN):
    """Cost of a gas year: annual base K, daily top-up to the forecast when the forecast exceeds K, overrun on the rest."""
    flow = np.asarray(flow, float)
    forecast = np.asarray(forecast, float)
    topup = np.maximum(forecast - base, 0.0)
    held = base + topup
    over = np.maximum(flow - held, 0.0)
    return float(base * len(flow) * c_a + topup.sum() * c_d + over.sum() * c_o)


def best_base(flow, forecast, grid=None, **kw):
    flow = np.asarray(flow, float)
    grid = np.linspace(0, flow.max() * 1.05, 120) if grid is None else grid
    costs = [cost(flow, k, forecast, **kw) for k in grid]
    i = int(np.argmin(costs))
    return float(grid[i]), float(costs[i])


def naive_forecast(s: pd.Series, lag=7):
    """Same weekday last week (a plain nomination-style forecast); the first week uses the running mean."""
    f = s.shift(lag)
    return f.fillna(s.expanding().mean())


def gas_year(idx):
    idx = pd.DatetimeIndex(idx)
    return np.where(idx.month >= 10, idx.year + 1, idx.year)

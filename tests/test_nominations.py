import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd

from nbpie import nominations as N


def _frame(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    flow = pd.Series(15.0 + np.sin(np.arange(n) / 5.0), index=idx)
    return pd.DataFrame({"moffat_physical_flow_mscm": flow, "moffat_nomination_exit_kwh": flow * 1e6 * 11.0})


def test_clean_series_has_no_flags():
    c = N.controls(_frame())
    assert not c["any_flag"].any()
    assert abs(c["cv"].median() - 11.0) < 1e-9


def test_seeded_defects_are_found():
    df = _frame()
    df.iloc[10, 1] = np.nan                      # missing nomination
    df.iloc[20, 1] = df.iloc[20, 1] * 2.3        # implausible calorific value
    df.iloc[30, 0] = 0.0                         # zero flow
    df.iloc[40, 1] = df.iloc[40, 1] * 1.15       # 15% deviation inside the CV band
    df = df.drop(df.index[50])                   # missing gas day
    c = N.controls(df)
    flagged = set(c.index[c["any_flag"]])
    for i in (10, 20, 30, 40):
        assert pd.date_range("2024-01-01", periods=60, freq="D")[i] in flagged
    assert pd.Timestamp("2024-02-20") in flagged
    assert c["missing_day"].sum() == 1


def test_imbalance_cost_is_proportional_to_error():
    e = pd.Series([1.0, -2.0], index=pd.date_range("2024-01-01", periods=2))
    price = pd.Series([40.0, 40.0], index=e.index)
    cost = N.imbalance_cost(e, price, share=0.05)
    assert np.isclose(cost.iloc[1], 2 * cost.iloc[0])
    assert np.isclose(cost.iloc[0], 1000 * 40 * 0.05)


def test_walk_forward_does_not_use_the_future():
    idx = pd.date_range("2023-01-01", periods=900, freq="D")
    y = pd.Series(np.arange(900, dtype=float), index=idx)
    X = pd.DataFrame({"x": np.arange(900, dtype=float)}, index=idx)

    class Mean:
        def fit(self, X, y): self.m = float(y.mean()); return self
        def predict(self, X): return np.full(len(X), self.m)

    p = N.walk_forward(y, X, Mean, "2024-06-01")
    first = p["2024-06-01"]
    assert np.isclose(first, y[:"2024-05-31"].mean())

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nbpie import booking, remit  # noqa: E402


def test_cost_flat_flow_books_base_equal_to_flow():
    flow = np.full(365, 10.0)
    k, c = booking.best_base(flow, flow)
    assert abs(k - 10.0) < 0.5
    assert abs(c - 10.0 * 365 * booking.C_ANNUAL) < 0.06 * c


def test_overrun_is_charged_when_nothing_is_booked():
    flow = np.full(10, 5.0)
    assert booking.cost(flow, 0.0, np.zeros(10)) == 5.0 * 10 * booking.C_OVERRUN


def test_topup_replaces_overrun_when_forecast_is_right():
    flow = np.array([10.0] * 5 + [20.0] * 5)
    c_topup = booking.cost(flow, 10.0, flow)
    c_none = booking.cost(flow, 10.0, np.full(10, 10.0))
    assert c_topup < c_none


def test_validator_finds_every_injected_error_and_nothing_else():
    df, injected = remit.synthetic_trades(errors=True)
    v = remit.validate(df, today="2027-01-01")
    found = set(zip(v.row, v.field))
    for r, f in injected.items():
        assert (r, f) in found
    assert {r for r in v.row} <= set(injected) | {14}   # the duplicate UTI is reported on both rows


def test_clean_file_has_no_violations():
    df, _ = remit.synthetic_trades(errors=False)
    assert remit.validate(df, today="2027-01-01").empty


def test_missing_column_reported():
    df, _ = remit.synthetic_trades(errors=False)
    assert (remit.validate(df.drop(columns=["price"]), today="2027-01-01").field == "price").any()


def test_gas_year():
    idx = pd.to_datetime(["2024-09-30", "2024-10-01", "2025-01-01"])
    assert list(booking.gas_year(idx)) == [2024, 2025, 2025]

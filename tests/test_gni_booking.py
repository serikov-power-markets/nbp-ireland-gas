import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nbpie import gni_booking as g  # noqa: E402


def test_published_examples_are_reproduced():
    # GNI example sheet: Moffat Entry monthly June 4.23 per MWh, daily June 0.21, daily January 8.70 (EUR per peak day MWh)
    assert abs(g.ANNUAL * g.MONTHLY_MULT[6] - 4.227009) < 1e-4
    assert abs(g.ANNUAL * g.DAILY_MULT[6] - 0.211350) < 1e-4
    assert abs(g.ANNUAL * g.DAILY_MULT[1] - 8.702665) < 1e-3


def test_flat_flow_books_annual():
    idx = pd.date_range("2025-10-01", "2026-09-30")
    flow = np.full(len(idx), 100.0)
    k, c = g.best_base(flow, flow, idx)
    assert abs(k - 100.0) < 8 and c < 1.2 * g.ANNUAL * 100.0


def test_overrun_charged_without_booking():
    idx = pd.date_range("2026-01-01", periods=3)
    c = g.cost(np.full(3, 10.0), np.zeros(3), idx, 0.0, overrun_mult=3.0)
    assert abs(c - 3 * 10 * g.ANNUAL * g.DAILY_MULT[1] * 3.0) < 1e-6

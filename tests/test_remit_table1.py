import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nbpie import remit_table1 as r  # noqa: E402


def test_field_list_is_complete():
    assert sorted(r.FIELDS) == list(range(1, 59))


def test_all_injected_defects_found_and_nothing_else():
    df, inj = r.synthetic(errors=True)
    v = r.validate(df, today="2027-01-01")
    found = {(int(a), b) for a, b in zip(v.row, v.field)}
    for row, f in inj.items():
        assert (row, f) in found
    assert {int(x) for x in v.row} <= set(inj) | {14}


def test_clean_file_has_no_findings():
    df, _ = r.synthetic(errors=False)
    assert r.validate(df, today="2027-01-01").empty

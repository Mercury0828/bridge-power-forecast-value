"""Phase 1e E0c: the TPIT schedule-error parser (experiments/p1e_tpit.py; docs/phase1e_plan.md v2, "Tests"):
- vintage selection: the latest snapshot dated <= A - h quarters that lists the project in Future;
- sign: eps = q(P) - q(A), so a project finished later than estimated has eps < 0;
- exclusions: actual dates after their snapshot's date, projects never completed, and projects not listed at the
  vintage (counted as not observed);
- the quarter grid q(d) = ceil((d - 2000-01-01) / 91.3125 days);
- snapshot selection: "_old" duplicates dropped, the "UPDATE" version preferred."""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from experiments.p1e_tpit import project_errors, qidx, select_files, snapshot_date, summarize  # noqa: E402

D = pd.Timestamp


def _fut(rows):
    return pd.DataFrame([dict(pid=p, proj=D(pr), snap=D(s), title="", desc="") for p, pr, s in rows])


def _comp(rows):
    return pd.DataFrame([dict(pid=p, actual=D(a), snap=D(s), title=t, desc="", to="TO") for p, a, s, t in rows])


def _by(rows):
    return {(r["pid"], r["h"]): r for r in rows}


FUT = _fut([("P1", "2019-06-01", "2018-02-01"), ("P1", "2019-12-01", "2018-06-01"), ("P1", "2020-03-01", "2019-06-01"),
            ("P4", "2019-01-01", "2017-02-01"), ("P5", "2019-01-01", "2016-02-01")])
COMP = _comp([("P1", "2020-06-15", "2020-10-01", "Line upgrade"), ("P1", "2020-06-15", "2021-02-01", "Line upgrade"),
              ("P2", "2021-05-01", "2021-02-01", "Future-dated actual"),
              ("P3", "2019-09-01", "2019-10-01", "x"), ("P3", "2019-08-01", "2020-02-01", "x"),
              ("P4", "2018-10-01", "2019-02-01", "New substation to serve data center load")])


def test_quarter_grid_is_ceiling():
    assert qidx(D("2000-01-01")) == 0
    assert qidx(D("2000-01-02")) == 1
    assert qidx(D("2024-01-01")) == 96          # 8,766 days = exactly 96 quarters of 91.3125 days
    assert qidx(D("2024-01-02")) == 97


def test_vintage_selection_and_sign():
    R = _by(project_errors(FUT, COMP))
    # h = 4: cutoff ~2019-06-16 -> the 2019-06-01 snapshot (P = 2020-03-01); late by one quarter
    assert R[("P1", 4)]["observed"] and R[("P1", 4)]["snap"] == "2019-06-01"
    assert R[("P1", 4)]["eps"] == qidx(D("2020-03-01")) - qidx(D("2020-06-15")) == -1
    # h = 8: cutoff ~2018-06-15 -> the 2018-06-01 snapshot (P = 2019-12-01)
    assert R[("P1", 8)]["snap"] == "2018-06-01" and R[("P1", 8)]["eps"] == -2
    # h = 12: no Future listing that early -> not observed
    assert not R[("P1", 12)]["observed"]
    # early completion: P4 estimated 2019-01-01, finished 2018-10-01 -> eps > 0 (the 91.3125-day grid is not the
    # calendar quarter: q = 77 and 75)
    assert R[("P4", 4)]["eps"] == qidx(D("2019-01-01")) - qidx(D("2018-10-01")) == 2


def test_exclusions_and_latest_actual():
    rows = project_errors(FUT, COMP)
    pids = {r["pid"] for r in rows}
    assert "P2" not in pids                     # its only actual date is after its snapshot's date
    assert "P5" not in pids                     # never completed
    R = _by(rows)
    assert R[("P3", 4)]["observed"] is False    # completed, never listed in Future: counted, not observed
    # P3's actual date comes from the latest snapshot listing it as completed
    comp3 = project_errors(_fut([("P3", "2019-01-01", "2018-06-01")]), COMP[COMP["pid"] == "P3"])
    assert {r["actual"] for r in comp3 if r["observed"]} == {"2019-08-01"}


def test_invalid_latest_record_excludes_the_project_without_fallback():
    """Plan v2 order: the latest Completed record is taken first, then validated. A
    placeholder date in the latest snapshot excludes the project; an earlier valid record is not used instead."""
    comp = _comp([("P6", "2018-01-10", "2019-06-01", "x"), ("P6", "9999-01-01", "2026-07-13", "x"),
                  ("P7", "2018-01-10", "2019-06-01", "x"), ("P7", None, "2026-07-13", "x")])
    fut = _fut([("P6", "2017-06-01", "2016-02-01"), ("P7", "2017-06-01", "2016-02-01")])
    assert project_errors(fut, comp) == []


def test_load_subset_and_summary_counts():
    rows = project_errors(FUT, COMP)
    R = _by(rows)
    assert R[("P4", 4)]["load"] and not R[("P1", 4)]["load"]
    S = summarize(rows)
    assert S["h4|all"]["n"] == 2 and S["h4|all"]["n_completed"] == 3 and S["h4|all"]["n_not_observed"] == 1
    assert S["h4|all"]["share_late"] == 0.5 and S["h4|all"]["share_early"] == 0.5
    assert S["h4|load"]["n"] == 1


def test_snapshot_selection(tmp_path):
    for name in ("ERCOT_June_TPIT_No_Cost_060118.xlsx", "ERCOT_June_TPIT_No_Cost_060118_old.xlsx",
                 "ERCOT October TPIT No Cost 100118.xlsx", "ERCOT October TPIT No Cost 100118 UPDATE.xlsx",
                 "notes.xlsx"):
        (tmp_path / name).write_bytes(b"")
    cur = tmp_path / "ERCOT July Ad-Hoc TPIT No Cost 071326 UPDATE.xlsx"
    cur.write_bytes(b"")
    files = select_files(tmp_path, cur)
    assert list(files) == [D("2018-06-01"), D("2018-10-01"), D("2026-07-13")]
    assert files[D("2018-06-01")].name == "ERCOT_June_TPIT_No_Cost_060118.xlsx"
    assert "UPDATE" in files[D("2018-10-01")].name
    assert snapshot_date("notes.xlsx") is None


# ---- E0-ext (Phase 1f; docs/phase1f_plan.md): the pre-2015 .xls rules --------------------------------------------------
def test_ext_snapshot_date_prefers_the_8_digit_group():
    from experiments.p1e_tpit import snapshot_date_ext
    assert snapshot_date_ext("ERCOT_MARCH_TPIT_No_Cost_03012012.xls") == D("2012-03-01")
    assert snapshot_date_ext("ERCOTDecemberTPITNoCost12012010_(2).xls") == D("2010-12-01")
    assert snapshot_date_ext("UPDATED__ERCOT_JUNE_TPIT_No_Cost_07292013.xls") == D("2013-07-29")
    assert snapshot_date_ext("ERCOT_FEBRUARY_TPITNoCost020114_022514.xls") == D("2014-02-01")   # 6-digit rule
    assert snapshot_date("ERCOT_MARCH_TPIT_No_Cost_03012012.xls") == D("2020-03-01")          # why the rule exists


def test_ext_file_selection_priority_and_exclusions(tmp_path):
    from experiments.p1e_tpit import select_files_ext
    for name in ("Comparison_Results_-_ERCOT_JUNE_TPIT_No_Cost_06012012.xls", "ERCOT_JUNE_TPIT_No_Cost_06012012.xls",
                 "updated_ERCOT_JUNE_TPIT_No_Cost_06012012.xls",
                 "Comparison_Results_ERCOTMarchTPITNoCost03012010.xls",
                 "ERCOT_November_TPIT_No_Cost_11012013_addendum.xls", "ERCOTTPIT1999_-_2008NoCost.xls",
                 "ERCOT-February-TPIT-No-Cost-020123.xlsx"):
        (tmp_path / name).write_bytes(b"")
    files = select_files_ext(tmp_path, before=D("2015-02-01"))
    assert list(files) == [D("2010-03-01"), D("2012-06-01")]
    assert files[D("2012-06-01")].name.startswith("updated_")                 # UPDATED > plain > comparison
    assert files[D("2010-03-01")].name.startswith("Comparison_Results")       # the only copy of that date


def test_ext_sheet_matching_is_case_insensitive_only_for_the_extension():
    from experiments.p1e_tpit import sheet_kind
    assert sheet_kind("FUTURETPIT12012011", case_insensitive=True) == "future"
    assert sheet_kind("COMPLETEDTPIT12012011", case_insensitive=True) == "completed"
    assert sheet_kind("FUTURETPIT12012011") is None                          # the primary rule, unchanged
    assert sheet_kind("FutureTPIT03012012NoCost") == "future"
    assert sheet_kind("CancelledTPIT03012012NoCost", case_insensitive=True) is None

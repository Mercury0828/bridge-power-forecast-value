"""Phase 1e E0c: a vintage-controlled ERCOT schedule-error family from the public TPIT archive (docs/phase1e_plan.md v2;
source S7 in docs/phase1e_sources.md v2). Fixed before any policy is evaluated under it.

    python experiments/p1e_tpit.py   -> data/p1e/tpit_errors.json, data/p1e/tpit_project_errors.csv

Data. ERCOT's archived Transmission Project and Information Tracking workbooks (2015-2026 .xlsx; about 3 snapshots
per year) plus the current workbook. They are downloaded to D:\\_caches\\ercot_tpit (not committed); the URLs and the
SHA-256 of the files are recorded.

Construction.
1. Snapshot date: from the file name (MMDDYY). Duplicate snapshots keep the "UPDATE" version and drop "_old".
2. Future sheets give (snapshot, ERCOT project number, projected in-service date P, title, description, status, TO).
   Completed sheets give the actual in-service date A. The record of the latest snapshot listing the project as
   completed is used. The project is excluded if that record's actual date is missing or after that snapshot's date;
   there is no fallback to an earlier record (plan v2, E0c step 1).
3. Vintage control. For horizon h quarters, take the latest snapshot dated <= A - h quarters in which the project is
   listed in Future with a projected date P.
   - The error is eps = q(P) - q(A), estimate minus actual, where q(d) = ceil((d - 2000-01-01) / 91.3125 days) (the
     model's quarterly grid; the model's sign convention: eps < 0 means the project finished later than estimated).
   - Projects not listed at that vintage are excluded for that horizon, and are counted.
4. Subsets: all projects (base), and a load-serving subset whose title or description matches LOAD_RE (declared here,
   before any evaluation).
5. Output: n, mean, SD, quantiles, the late/on-time/early shares, and the pmf clipped to [-16, 16], per (h, subset).
   The per-project table is written too.
Survivorship: cancelled projects are excluded.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import sys
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "p1e"
CACHE = pathlib.Path(r"D:\_caches\ercot_tpit")
ARCHIVE_URL = "https://www.ercot.com/files/docs/2021/10/22/Archived-Transmission-Project-and-Information-Tracking.zip"
CURRENT_URL = "https://www.ercot.com/files/docs/2022/03/02/ERCOT-July-Ad-Hoc-TPIT-No-Cost-071326-UPDATE.xlsx"
HORIZONS = (4, 8, 12)
LOAD_RE = re.compile(r"\b(load|loads|customer|customers|data ?cent(?:er|re)s?|serve|serves|serving)\b", re.I)
QDAYS = 91.3125
ORIGIN = pd.Timestamp("2000-01-01")
COLS = {"pid": "ERCOT Project Number", "title": "Project Title", "desc": "Project Description",
        "status": "Transmission Status", "to": "Transmission Owner (text", "proj": "Projected In-Service Date",
        "actual": "Actual In-Service Date"}


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def fetch():
    CACHE.mkdir(parents=True, exist_ok=True)
    z, cur = CACHE / "archive.zip", CACHE / "current.xlsx"
    for url, f in ((ARCHIVE_URL, z), (CURRENT_URL, cur)):
        if not f.exists():
            urllib.request.urlretrieve(url, f)
    x = CACHE / "x"
    if not x.exists():
        with zipfile.ZipFile(z) as zz:
            zz.extractall(x)
    return dict(archive=dict(url=ARCHIVE_URL, sha256=_sha(z)), current=dict(url=CURRENT_URL, sha256=_sha(cur))), x, cur


def snapshot_date(name):
    m = re.search(r"(\d{2})(\d{2})(\d{2})", name)
    return pd.Timestamp(2000 + int(m.group(3)), int(m.group(1)), int(m.group(2))) if m else None


def select_files(x, cur):
    files = {}
    for f in sorted(list(x.glob("*.xlsx")) + [cur]):
        d = snapshot_date(f.name)
        if d is None or "old" in f.name.lower():
            continue
        prev = files.get(d)
        if prev is None or ("UPDATE" in f.name.upper() and "UPDATE" not in prev.name.upper()):
            files[d] = f
    return dict(sorted(files.items()))


def read_sheet(xl, sheet):
    raw = xl.parse(sheet, header=None)
    hdr = next(i for i in range(min(10, len(raw)))
               if any(str(v).strip().startswith(COLS["pid"]) for v in raw.iloc[i].tolist()))
    names = [str(v).strip() for v in raw.iloc[hdr].tolist()]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = names
    out = {}
    for key, prefix in COLS.items():
        col = next((c for c in names if c.startswith(prefix)), None)
        out[key] = df[col] if col is not None else pd.Series([None] * len(df), index=df.index)
    t = pd.DataFrame(out)
    t["pid"] = t["pid"].astype(str).str.strip()
    t = t[(t["pid"] != "nan") & (t["pid"] != "")]
    for c in ("proj", "actual"):
        t[c] = pd.to_datetime(t[c], errors="coerce")
    return t


def qidx(d):
    """The quarter of a date on the model's quarterly grid: ⌈(d − ORIGIN) / QDAYS⌉ (plan v2, E0c step 1)."""
    return int(np.ceil((d - ORIGIN).days / QDAYS))


def project_errors(fut, comp):
    """Vintage-controlled errors from stacked Future rows (pid, proj, snap, title, desc) and Completed rows (pid, actual,
    snap, title, desc, to). -> one row per (completed project, horizon)."""
    # plan v2 order: take the record of the latest snapshot listing the project as completed, then exclude it if
    # its actual date is missing or after that snapshot's date (no fallback to an earlier record)
    latest = comp.sort_values("snap", kind="stable").groupby("pid").tail(1)
    actual = latest[latest["actual"].notna() & (latest["actual"] <= latest["snap"])].set_index("pid")
    rows = []
    fut_ok = fut[fut["proj"].notna()]
    for pid, a in actual.iterrows():
        A = a["actual"]
        hist = fut_ok[fut_ok["pid"] == pid]
        text = f"{a.get('title', '')} {a.get('desc', '')}"
        load = bool(LOAD_RE.search(text))
        for h in HORIZONS:
            cutoff = A - pd.Timedelta(days=h * QDAYS)
            cand = hist[hist["snap"] <= cutoff]
            if cand.empty:
                rows.append(dict(pid=pid, h=h, observed=False, load=load, actual=A.date().isoformat()))
                continue
            c = cand.sort_values("snap", kind="stable").iloc[-1]
            rows.append(dict(pid=pid, h=h, observed=True, load=load, to=str(a.get("to", ""))[:20],
                             title=str(a.get("title", ""))[:60], actual=A.date().isoformat(),
                             snap=c["snap"].date().isoformat(), proj=c["proj"].date().isoformat(),
                             eps=qidx(c["proj"]) - qidx(A)))
    return rows


def build():
    prov, x, cur = fetch()
    files = select_files(x, cur)
    fut, comp = [], []
    for d, f in files.items():
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            if "uture" in sh:
                t = read_sheet(xl, sh)
                t["snap"] = d
                fut.append(t)
            elif "omplet" in sh:
                t = read_sheet(xl, sh)
                t["snap"] = d
                comp.append(t)
    rows = project_errors(pd.concat(fut, ignore_index=True), pd.concat(comp, ignore_index=True))
    return prov, files, rows, max(files)


def summarize(rows):
    out = {}
    for h in HORIZONS:
        for subset in ("all", "load"):
            R = [r for r in rows if r["h"] == h and (subset == "all" or r["load"])]
            obs = [r for r in R if r["observed"]]
            e = np.array([r["eps"] for r in obs], dtype=float)
            if len(e) == 0:
                continue
            ec = np.clip(e, -16, 16)
            pmf = {int(k): float((ec == k).mean()) for k in range(-16, 17) if (ec == k).any()}
            out[f"h{h}|{subset}"] = dict(
                n=int(len(e)), n_completed=int(len(R)), n_not_observed=int(len(R) - len(obs)),
                mean=float(e.mean()), sd=float(e.std(ddof=1)) if len(e) > 1 else 0.0,
                quantiles={q: float(np.quantile(e, q / 100)) for q in (5, 25, 50, 75, 95)},
                share_late=float((e < 0).mean()), share_on_time=float((e == 0).mean()), share_early=float((e > 0).mean()),
                share_clipped=float((np.abs(e) > 16).mean()), pmf=pmf)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prov, files, rows, last_snap = build()
    S = dict(provenance=dict(prov, n_snapshots=len(files), first=str(min(files).date()), last=str(last_snap.date()),
                             files=[f.name for f in files.values()], load_regex=LOAD_RE.pattern,
                             quarter_days=QDAYS, horizons=HORIZONS),
             summary=summarize(rows))
    (OUT / "tpit_errors.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    pd.DataFrame([r for r in rows if r["observed"]]).to_csv(OUT / "tpit_project_errors.csv", index=False)
    for k, v in S["summary"].items():
        print(f"{k:10s} n {v['n']:4d} (not observed {v['n_not_observed']:4d}) mean {v['mean']:+.2f} sd {v['sd']:.2f} "
              f"q50 {v['quantiles'][50]:+.1f} late {v['share_late']:.2f} on {v['share_on_time']:.2f} "
              f"early {v['share_early']:.2f}")


# ---- E0-ext (Phase 1f; docs/phase1f_plan.md "E0-ext"; owner D-026): the pre-2015 .xls workbooks ---------------------
# A pre-declared sensitivity. The primary construction above is unchanged, and its outputs are not touched.
def snapshot_date_ext(name):
    """An 8-digit MMDDYYYY group if present (e.g. 03012012 -> 2012-03-01), otherwise the primary 6-digit MMDDYY rule."""
    m = re.search(r"(?<!\d)(\d{2})(\d{2})(\d{4})(?!\d)", name)
    if m and 1990 <= int(m.group(3)) <= 2035 and 1 <= int(m.group(1)) <= 12 and 1 <= int(m.group(2)) <= 31:
        return pd.Timestamp(int(m.group(3)), int(m.group(1)), int(m.group(2)))
    return snapshot_date(name)


def _ext_priority(name):
    n = name.lower()
    return 0 if "update" in n else (2 if "compar" in n else 1)


def select_files_ext(x, before):
    """The pre-2015 .xls snapshots: the cumulative 1999-20xx files and addenda are excluded; one file per snapshot date,
    preferring names with UPDATED, then plain names, then comparison-report copies, then the first name."""
    cand = {}
    for f in sorted(x.glob("*.xls")):
        n = f.name.lower()
        if not n.endswith(".xls") or "1999" in n or "addendum" in n:
            continue
        d = snapshot_date_ext(f.name)
        if d is None or d >= before:
            continue
        cand.setdefault(d, []).append(f)
    return {d: sorted(fs, key=lambda f: (_ext_priority(f.name), f.name))[0] for d, fs in sorted(cand.items())}


def sheet_kind(sheet, case_insensitive=False):
    s = sheet.lower() if case_insensitive else sheet
    return "future" if "uture" in s else ("completed" if "omplet" in s else None)


def load_sheets(files, case_insensitive, require_both=False):
    """-> (future tables, completed tables, failures). A workbook that cannot be opened, or that yields no parseable
    Future and Completed sheet, is excluded and listed. A single unparseable sheet is listed as "file:sheet"."""
    fut, comp, failed = [], [], []
    for d, f in files.items():
        try:
            xl = pd.ExcelFile(f)
        except Exception as e:                          # noqa: BLE001
            failed.append(f"{f.name}: cannot open ({type(e).__name__})")
            continue
        kinds, tabs = set(), []
        for sh in xl.sheet_names:
            kind = sheet_kind(sh, case_insensitive)
            if kind is None:
                continue
            try:
                t = read_sheet(xl, sh)
            except StopIteration:                       # no header row found
                failed.append(f"{f.name}:{sh}: no header")
                continue
            t["snap"] = d
            tabs.append((kind, t))
            kinds.add(kind)
        if require_both and kinds != {"future", "completed"}:        # pre-2015 files only; the primary set is as is
            failed.append(f"{f.name}: excluded (parsed sheets {sorted(kinds)})")
            continue
        for kind, t in tabs:
            (fut if kind == "future" else comp).append(t)
    return fut, comp, failed


def build_ext():
    prov, x, cur = fetch()
    prim = select_files(x, cur)
    ext = select_files_ext(x, before=min(prim))
    f1, c1, _ = load_sheets(prim, case_insensitive=False)
    f2, c2, failed = load_sheets(ext, case_insensitive=True, require_both=True)
    rows = project_errors(pd.concat(f1 + f2, ignore_index=True), pd.concat(c1 + c2, ignore_index=True))
    return prov, prim, ext, failed, rows


def main_ext():
    OUT.mkdir(parents=True, exist_ok=True)
    prov, prim, ext, failed, rows = build_ext()
    era = {"completed_before_2018": summarize([r for r in rows if r["actual"] < "2018"]),
           "completed_2018_on": summarize([r for r in rows if r["actual"] >= "2018"])}
    S = dict(provenance=dict(prov, rules="docs/phase1f_plan.md E0-ext (pre-declared)", primary_snapshots=len(prim),
                             ext_snapshots=len(ext), ext_dates=[str(d.date()) for d in ext],
                             ext_files=[f.name for f in ext.values()], ext_failed_to_parse=failed,
                             load_regex=LOAD_RE.pattern, quarter_days=QDAYS, horizons=HORIZONS),
             summary=summarize(rows), era=era)
    (OUT / "tpit_errors_ext.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
    pd.DataFrame([r for r in rows if r["observed"]]).to_csv(OUT / "tpit_project_errors_ext.csv", index=False)
    print(f"ext snapshots {len(ext)} (failed {failed}); primary {len(prim)}")
    for k, v in S["summary"].items():
        print(f"{k:10s} n {v['n']:4d} (not observed {v['n_not_observed']:4d}) mean {v['mean']:+.2f} sd {v['sd']:.2f}")


if __name__ == "__main__":
    import sys as _sys
    main_ext() if "--extended" in _sys.argv[1:] else main()

# The value of grid-connection information in bridge-power planning for AI data centres — code and results

Jiachen Shen (University of Houston) and Hui Zhong (Miami University)

This repository holds the model, the run definitions, the analysis code and the results behind the paper
*The value of grid-connection information in bridge-power planning for AI data centres* (submitted to *Energy*).

An AI data-centre campus of 150 MW of IT load plans on-site bridge power before its grid-connection quarter is known:
gas CHP gensets with absorption chillers, diesel gensets, rented gensets, a battery and purchased standby, each with a
procurement lead time. Midway through planning the utility issues an estimated energization quarter. The model is a
multistage stochastic mixed-integer linear program solved with HiGHS; plans with and without the forecast are
evaluated exactly under declared joint laws of the connection quarter and the forecast.

## Layout

| Path | Contents |
|---|---|
| `model/` | `bridge.py` (parameters, the multistage MILP, cost recomputation, capacity checks); `signal.py` (one plan copy per forecast category, linked before the forecast); `evidence.py` (connection-time laws and report outcomes) |
| `experiments/` | run definitions (`p1c_core.py`, `p1d_core.py`), the resumable runner (`harness.py`), the recorded solver repairs (`p1d_repair.py`), analyses (`analyze_*.py`), report and table generators (`make_*_report.py`), figures (`make_figures.py`) and run chains (`run_*.sh`) |
| `data/` | summaries (`*_summary.json`), cell rows (`*_rows.json`), generated tables (`*_tables.md`, `p1g_result_tables.md`), certificate outputs (`gate*_*.json`), TPIT-derived errors (`p1e/`), the run manifest |
| `tests/` | `pytest` suite: bookkeeping, duality and brute-force checks, the forecast tree, the variants |
| `tools/` | `md2tex_tables.py` and `decomp_table.py`, which turn the tables in `data/` into the supplementary tables |

The job-level results of every run (one JSON item per solved plan, the run plans and the logs, 21,130 files) are in
the archive `runs.zip` at the repository root. Unzip it into `experiments/` to obtain `experiments/r2_runs/`.

## Environment

- Python 3.11. Install the pinned packages with `pip install -r requirements.txt` (key pins: highspy 1.15.1,
  numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2).
- On Windows set `PYTHONIOENCODING=utf-8`.
- Every solve is deterministic: HiGHS runs single-threaded and the seeds are fixed in the run definitions.

## Reproducing the paper

From the stored runs (minutes):

```bash
python -m pytest tests -q
python experiments/analyze_p1c.py --run p1c_main          # registered benchmark: data/p1c_summary.json, p1c_rows.json
python experiments/make_p1c_report.py --run p1c_main      # data/p1c_tables.md
python experiments/analyze_p1d.py --variant spine --run p1d_spine   # one exploratory variant (see data/run_manifest.md)
python experiments/make_p1d_report.py                     # data/p1d_tables.md (also make_p1e_report.py, make_p1f_report.py)
python experiments/make_p1g_report.py                     # data/p1g_result_tables.md (rental carry-over, gas rentals)
python experiments/make_figures.py --paper                # the paper's figures, into paper/figs/
python tools/md2tex_tables.py <spec.json> <out.tex>       # tables in LaTeX
```

`python reproduce_check.py` runs all of these analyses and report generators at once. It then compares every
regenerated file in `data/` with its released version, line endings normalized. Set `REF_FIGS` to a directory of
reference renders to compare the figures pixel by pixel as well.

From scratch (hours; 16 parallel workers recommended):

```bash
python experiments/harness.py --kind p1c --run p1c_main --workers 16 --time-limit 1800
python experiments/harness.py --kind p1d --variant spine --run p1d_spine --workers 16 --time-limit 1800
python experiments/p1d_repair.py --run p1d_spine          # recorded solver rules; a second call applies the escalation
bash experiments/run_p1f_chain.sh                         # all Phase-1f variants
bash experiments/run_p1g_chain.sh                         # rental carry-over and gas rentals
python experiments/p1e_tpit.py && python experiments/p1e_eval.py   # downloads the public ERCOT TPIT workbooks
```

## Where each result comes from

| Paper item | Data | Code |
|---|---|---|
| Table 4, Fig. 3 (registered benchmark) | `data/p1c_summary.json`, `data/p1c_tables.md` | `analyze_p1c.py`, `make_p1c_report.py`, `make_figures.py` (`fig_main_values`) |
| Table 4 right columns, Fig. 5 (implementation) | `data/p1d/{spine,units,e2,e4b_spine,e1,...}_summary.json` | `analyze_p1d.py`, `make_figures.py` (`fig_implementation`) |
| Fig. 4 (representative cell) | `r2_runs/p1c_main/items/` | `make_figures.py` (`fig_policy_illustration`) |
| Fig. 6 (techno-economic levels) | `data/p1d/s_*_summary.json` | `make_p1f_report.py`, `make_figures.py` (`fig_tornado`) |
| Fig. 7, Fig. 8 (face value, timing) | `data/p1d/{naive,tau2,tau6}_summary.json` | `make_figures.py` (`fig_naive`, `fig_value_vs_tau`) |
| Table 5 (physical quantities) | `data/p1d/representative_cells.json` | `p1d_accounting.py` |
| Rental carry-over, gas rentals | `data/p1d/{carry,gasrent_*}_summary.json`, `data/p1g_result_tables.md` | `make_p1g_report.py` |
| TPIT schedule errors | `data/p1e/tpit_*.json`, `data/p1e/*.csv` | `p1e_tpit.py`, `p1e_eval.py` |
| Certificates (Supplementary S3) | `data/gate2_vertex_certificates_*.json` | `gate2_vertex_certificates.py` |

Code comments cite planning documents and decision records (`docs/...`, `D-0xx`) that are not part of this release;
the pre-registration and amendment records are available from the corresponding author on request.

## Citation

Please cite the paper and this repository (`CITATION.cff`). The repository is archived on Zenodo:
[doi:10.5281/zenodo.22986729](https://doi.org/10.5281/zenodo.22986729) (all versions).

## Licence

Code: MIT (`LICENSE`). Data and results, including `runs.zip`: CC BY 4.0 (`LICENSE-data`).

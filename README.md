# Bridge-power procurement for AI data centres: the value of an imperfect grid-connection date estimate — code and results

Jiachen Shen (University of Houston) and Hui Zhong (Miami University)

This repository holds the model, the run definitions, the analysis code and the results behind the paper
*Bridge-power procurement for AI data centres: the value of an imperfect grid-connection date estimate*. An earlier
version of the paper had the title *The value of grid-connection information in bridge-power planning for AI data
centres*; the sections "Reproducing the paper" and "Where each result comes from" below refer to that version's
numbering, and the section "Current paper" maps the current tables and figures.

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
The runs of the face-value fallback and resale-value checks (Sections 5.4 and 5.5 of the current paper; 1,286 files)
are in `runs_es.zip`; unzip it into `experiments/` in the same way.

## Environment

- Python 3.11. Install the pinned packages with `pip install -r requirements.txt` (key pins: highspy 1.15.1,
  numpy 2.4.6, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2).
- On Windows set `PYTHONIOENCODING=utf-8`.
- Every solve is deterministic: HiGHS runs single-threaded and the seeds are fixed in the run definitions.

## Reproducing the paper

From the stored runs (minutes):

```bash
python -m pytest tests -q
python experiments/analyze_p1c.py --run p1c_main          # benchmark: data/p1c_summary.json, p1c_rows.json
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
| Table 4, Fig. 3 (benchmark) | `data/p1c_summary.json`, `data/p1c_tables.md` | `analyze_p1c.py`, `make_p1c_report.py`, `make_figures.py` (`fig_main_values`) |
| Table 4 right columns, Fig. 5 (implementation) | `data/p1d/{spine,units,e2,e4b_spine,e1,...}_summary.json` | `analyze_p1d.py`, `make_figures.py` (`fig_implementation`) |
| Fig. 4 (representative cell) | `r2_runs/p1c_main/items/` | `make_figures.py` (`fig_policy_illustration`) |
| Fig. 6 (techno-economic levels) | `data/p1d/s_*_summary.json` | `make_p1f_report.py`, `make_figures.py` (`fig_tornado`) |
| Fig. 7, Fig. 8 (face value, timing) | `data/p1d/{naive,tau2,tau6}_summary.json` | `make_figures.py` (`fig_naive`, `fig_value_vs_tau`) |
| Table 5 (physical quantities) | `data/p1d/representative_cells.json` | `p1d_accounting.py` |
| Rental carry-over, gas rentals | `data/p1d/{carry,gasrent_*}_summary.json`, `data/p1g_result_tables.md` | `make_p1g_report.py` |
| TPIT schedule errors | `data/p1e/tpit_*.json`, `data/p1e/*.csv` | `p1e_tpit.py`, `p1e_eval.py` |
| Certificates (Supplementary S3) | `data/gate2_vertex_certificates_*.json` | `gate2_vertex_certificates.py` |

## Current paper

The current version adds four computations to the stored runs. None of them changes a stored result.

```bash
python experiments/es_perfect_info.py     # perfect-information reference: 116 deterministic path solves (minutes)
python experiments/w3_decision_map.py     # decision map: re-tabulates stored rows and policies, no solve
python experiments/es_model_size.py       # model sizes (builds, does not solve) and stored solve times
python experiments/make_figures_es.py     # result figures at the 119 mm print width, into paper_es/figs/
python experiments/es_fallback_resale.py  # face-value fallback and resale value: tables from the runs in runs_es.zip
```

The face-value fallback and resale-value checks are two new variants of `experiments/p1d_core.py`: `naive_eps` (the
face-value planner with weight 0.01 or 0.05 on the broad planner kernel, `p1c_core.eps_kernel`; 280 jobs) and `s_salv`
(all resale fractions of new equipment scaled by 0.5 and 1.5; 352 jobs). `bash experiments/run_es_fallback_resale.sh`
solves them from scratch (hours), applies the recorded solver rules, analyses them and writes the tables.

`model/bridge.py` gains the option `only_T=t`, which builds the deterministic path model that knows the connection
quarter `t` from the start (same procurement rules, only branch `t`). The default model is unchanged.

| Current paper item | Data | Code |
|---|---|---|
| Fig. 1 (information and commitments), Fig. 2 (campus energy system) | — | `paper_es/figs/fig_timeline.tex`, `paper_es/figs/fig_system.tex` (TikZ, `pdflatex`) |
| Table 4 (techno-economic data), Table 5 (main variants) | — | parameter values in `model/bridge.py` (`Params`); variant definitions in `experiments/p1d_core.py` (`VARIANTS`) |
| Table 6 (value, allowance and threshold columns), Fig. 3 (value of the estimate) | `data/p1c_summary.json`, `data/p1c_tables.md` | `analyze_p1c.py`; `make_figures_es.py` (`main_values`) for Fig. 3 |
| Table 6, columns LC, RU and Joint | `data/p1d/spine_summary.json` (LC), `data/p1d/units_summary.json` (RU), `data/p1d/e4b_spine_summary.json` (Joint) | `analyze_p1d.py` |
| Table 7 (perfect-information reference) | `data/es_perfect_info.json` | `es_perfect_info.py` |
| Table 8 (decision map), Online Resource 1 decision-map tables | `data/w3/decision_map.json`, `data/w3/decision_map.md` | `w3_decision_map.py` |
| Fig. 4 (running example) | `experiments/r2_runs/p1c_main/items/` | `make_figures_es.py` (`policy_illustration`) |
| Fig. 5 (face value), Fig. 6 (techno-economic levels), Fig. 7 (timing) | `data/p1d/{naive,s_*,tau2,tau6}_summary.json` (Fig. 6 does not use `s_salv`) | `make_figures_es.py` (`naive`, `tornado`, value against arrival quarter) |
| Table 9 (physical quantities) | `data/p1d/representative_cells.json` | `p1d_accounting.py repcells` |
| Computation paragraph (model size, solve times) | `data/es_compute.json` | `es_model_size.py` |
| Section 5.4 (face-value fallback), Online Resource 1 Tables S49–S50 | `data/p1d/naive_eps_{rows,summary}.json`, `data/es_fallback_resale.json`, `data/es_fallback_resale.md` | `analyze_p1d.py --variant naive_eps --run es_naive_eps`; `es_fallback_resale.py` |
| Section 5.5 (resale value), Online Resource 1 Tables S51–S52 | `data/p1d/s_salv_{rows,summary}.json`, `data/es_fallback_resale.json`, `data/es_fallback_resale.md` | `analyze_p1d.py --variant s_salv --run es_s_salv`; `es_fallback_resale.py` |

Run, file and field names keep the stage labels of the computations: p1c (Phase 1c) is the benchmark analysis, and the later labels p1d to p1g (Phase 1d to 1g) mark the variants and checks built on it, so a field such as `vs_phase1c` in a variant summary compares the variant with the benchmark. Some code comments cite design notes that are not part of this release; they are not needed to run the code or to reproduce the results.

## Citation

Please cite the paper and this repository (`CITATION.cff`). The repository is archived on Zenodo:
[doi:10.5281/zenodo.22986729](https://doi.org/10.5281/zenodo.22986729) (all versions).

## Licence

Code: MIT (`LICENSE`). Data and results, including `runs.zip` and `runs_es.zip`: CC BY 4.0 (`LICENSE-data`).

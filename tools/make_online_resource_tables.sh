#!/usr/bin/env bash
# The LaTeX result tables S2-S39 of Online Resource 1 from the released files, into build/tables/.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
mkdir -p build/tables
"$PY" experiments/make_supplement_tables.py
"$PY" tools/decomp_table.py build/tables/tables_decomp_s3.tex
"$PY" tools/md2tex_tables.py tools/online_resource_tables_s3.json build/tables/tables_part_s3.tex
"$PY" tools/md2tex_tables.py tools/online_resource_tables_s4.json build/tables/tables_part_s4.tex

# MESSY_synth

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests](https://github.com/mmcdermott/MESSY_synth/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/MESSY_synth/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/mmcdermott/MESSY_synth/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/MESSY_synth/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/mmcdermott/MESSY_synth.svg)](https://github.com/mmcdermott/MESSY_synth/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/MESSY_synth/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/MESSY_synth#license)

**Generate synthetic raw data that matches the shape a [MEDS-Extract](https://github.com/mmcdermott/MEDS_extract)
v0.7 MESSY config expects — then run the whole ETL over it, with no access to the real dataset.**

A MESSY file is a complete, config-only ETL. It declares where the raw data comes from and how each
raw table becomes MEDS events, written in the [dftly](https://github.com/mmcdermott/dftly)
expression DSL. What it never declares is what the raw tables *look like* — those live behind a
credentialed download.

MESSY_synth runs that reasoning backwards. It reads the config's expressions, deduces what the
source files must have contained for those expressions to make sense, and writes a directory of
synthetic files with the right filenames, columns, dtypes, string formats, and keys.

That gets you two things:

- **A runnable demo of any ETL**, with no credentials and no data-use agreement.
- **A smoke test**, so a change to a MESSY config can be checked in CI: regenerate, run, and fail
    the build if any event stopped producing rows.

> [!IMPORTANT]
> Every generated value is deliberately, visibly fake — `SYNTH_ITEMID_003`, not a real LOINC code.
> The goal is to reproduce a dataset's **structure**, never its content. Nothing in the output is
> derived from, or resembles, any real source dataset or patient.

## Install

Not yet published to PyPI — install from source:

```bash
uv add git+https://github.com/mmcdermott/MESSY_synth
```

Or for development:

```bash
git clone https://github.com/mmcdermott/MESSY_synth
cd MESSY_synth
uv sync --group dev
uv run pre-commit install
```

## Quick start

```bash
# Generate synthetic sources for an ETL, then look at what it inferred.
MESSY-synth path/to/messy.yaml -o raw_synthetic/ --explain

# Run the real ETL over them.
meds-extract-run spec=path/to/messy.yaml output_dir=meds_out/ \
	do_download=false input_dir=raw_synthetic/
```

From Python, `synthesize` does the inference, generation, writing, and validation in one call:

```python
>>> from MEDS_extract.config import MessyConfig
>>> from MESSY_synth import synthesize
>>> cfg = MessyConfig.parse({
...     "etl": {"dataset_name": "Demo", "raw_dataset_version": "1"},
...     "patients": {
...         "_defaults": {"subject_id": "$pid"},
...         "dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'},
...         "sex": {"code": 'f"SEX//{$sex}"', "time": None},
...     },
...     "labs": {
...         "_defaults": {"subject_id": "$pid"},
...         "lab": {
...             "code": 'f"LAB//{$itemid}"',
...             "time": '$ts::"%Y-%m-%d %H:%M:%S"',
...             "numeric_value": "$value",
...             "_metadata": {"d_labitems": {"itemid": "$itemid", "description": "$label"}},
...         },
...     },
... })
>>> with tempfile.TemporaryDirectory() as d:
...     result = synthesize(cfg, d, seed=0, n_subjects=10, rows_per_subject=2)
...     print_directory(Path(d))
├── _MESSY_synth_manifest.json
├── d_labitems.csv
├── labs.csv
└── patients.csv

```

Three files, because the config mentions three: two event tables and the `_metadata` dictionary
that `labs` looks its codes up in. The generated `labs` table carries exactly the columns the
config reads, with `ts` rendered in the format the config parses it with:

```python
>>> with tempfile.TemporaryDirectory() as d:
...     result = synthesize(cfg, d, seed=0, n_subjects=10, rows_per_subject=2)
...     result.dataset.frames["labs"].head(3)
shape: (3, 4)
┌─────┬──────────────────┬─────────────────────┬────────┐
│ pid ┆ itemid           ┆ ts                  ┆ value  │
│ --- ┆ ---              ┆ ---                 ┆ ---    │
│ i64 ┆ str              ┆ str                 ┆ f64    │
╞═════╪══════════════════╪═════════════════════╪════════╡
│ 1   ┆ SYNTH_ITEMID_000 ┆ 2011-01-28 14:10:11 ┆ 73.623 │
│ 2   ┆ SYNTH_ITEMID_005 ┆ 2013-02-17 19:59:30 ┆ 54.308 │
│ 3   ┆ SYNTH_ITEMID_001 ┆ 2012-06-21 15:58:27 ┆ 73.642 │
└─────┴──────────────────┴─────────────────────┴────────┘

```

## What it guarantees

Structural validity is not a per-column property. A MEDS-Extract run only yields a real dataset if
values line up *across* files, and every one of those relationships is silent when it breaks — a
dangling join key nulls `subject_id` and the row is dropped with no error at all.

So the generator maintains four invariants:

| Invariant                                                                  | Why it matters                                                                     |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Every table's subject column draws from **one shared subject universe**    | Otherwise events never merge onto shared subjects                                  |
| Every join key **exists on the far side**                                  | MEDS-Extract joins are left joins; a dangling key silently deletes the row         |
| Every `_metadata` dictionary **covers its event's code vocabulary**        | Otherwise the metadata join matches nothing and descriptions come back null        |
| Each subject gets a **coherent timeline** — birth, then events, then death | MEDS treats birth/death as record bounds, and downstream tools assume the ordering |

The first three are one mechanism: a union-find over `(table, column)` pairs groups every column
that must agree, and each group gets a single shared value pool.

## How it infers types

Expectations flow *down* the dftly AST from each MEDS output slot — `time` must yield a timestamp,
`numeric_value` a number, `code` a string — and each node says how its own expectation constrains
its children. Column references are not always leaves: a `_table.cols` name is expanded in place,
and a name pulled in by `_table.join` is attributed to the table it actually lives in.

```python
>>> from MESSY_synth import infer_constraints
>>> cs = infer_constraints(cfg)
>>> for col in ("pid", "ts", "itemid", "value"):
...     c = cs.get("labs", col)
...     print(f"{col:<8} {c.kind.name:<13} {c.datetime_formats}")
pid      SUBJECT_ID    ()
ts       DATETIME_STR  ('%Y-%m-%d %H:%M:%S',)
itemid   CATEGORICAL   ()
value    NUMERIC       ()

```

Some of what it deduces is less obvious than a dtype:

- **`strptime` formats** become the literal characters written to disk. A column parsed with
    `coalesce($x::?"%Y-%m-%d %H:%M:%S", $x::?"%Y-%m-%d")` gets rows in *both* formats, so both
    branches of the coalesce are exercised.
- **Compared-against literals** are seeded into the value pool *and* fix the column's type.
    `$admissioncount == 1` means the column is an integer that must sometimes equal 1 — otherwise the
    branch it gates never fires, and generating a string there makes polars raise outright.
- **Regexes are solved, not ignored.** AmsterdamUMCdb derives every timestamp in its `admissions`
    table from `extract /2003|2010/ from $admissionyeargroup`. Opaque tokens there yield null and
    delete every event in the table, so `MESSY_synth` generates conforming values — and, for regexes
    that merely *test* a column, a mix of matching and non-matching ones.
- **Magnitudes implied by arithmetic.** `($anchor_year - $anchor_age)::str` parsed as a year only
    works if the first term looks like a calendar year. The default 1–100 range would make every
    birth event vanish.
- **Birth and death roles.** A column feeding a `MEDS_BIRTH` or `MEDS_DEATH` timestamp is placed on
    the subject's timeline rather than sampled.

`--explain` prints all of it, per column, with the reasoning attached.

## Smoke-testing an ETL

`smoke_test` generates, runs `meds-extract-run`, and then checks the *artifacts* — not just the
exit status. That distinction is the point: MEDS-Extract exits 0 when a lenient time cast nulls
every row of a table, and exits 0 when re-run into a populated output directory without executing a
single stage.

```python
from MESSY_synth import smoke_test


def test_etl_still_works(tmp_path):
    result = smoke_test("src/MY_MEDS/configs/messy.yaml", tmp_path)
    assert result.ok, result.summary()
```

The CLI exits non-zero on any error finding, so it works directly in CI:

```bash
MESSY-synth src/MY_MEDS/configs/messy.yaml -o "$RUNNER_TEMP/raw" || exit 1
```

## Validated against real ETLs

`demo/smoke_matrix.py` runs the whole pipeline across a set of configs. Against the v0.7 ETL
configs available at the time of writing:

| ETL                  | Result                 | Output from synthetic input                                                                     |
| -------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| MEDS-Extract example | pass                   | 185 events, 16 subjects, 64 codes (40 described)                                                |
| MIMIC-IV             | pass                   | 2,827 events, 40 subjects, 635 codes (312 described)                                            |
| NWICU                | pass                   | 1,447 events, 40 subjects, 275 codes (130 described)                                            |
| SICdb                | pass                   | 851 events, 40 subjects, 250 codes                                                              |
| AmsterdamUMCdb       | pass                   | 1,294 events, 40 subjects, 711 codes                                                            |
| HiRID                | **fails — config bug** | `raw_stage/observation_tables.datetime` is read both via `strptime` and directly as a timestamp |
| INSPIRE              | **fails — config bug** | `operations` declares a self-join, which MEDS-Extract 0.7 rejects                               |

The last two are the tool working as intended. Neither config can run on *any* input, real or
synthetic, and MESSY_synth reports each with the specific column or table at fault rather than a
stack trace. eICU has no v0.7 MESSY config yet, so it is not represented.

## Limitations

- **Values are structurally valid, not clinically plausible.** A numeric column defaults to the
    range 1–100 unless the config implies otherwise; use `-R col=lo:hi` where that matters.
- **Only regex syntax that can be run backwards is supported.** Lookarounds and backreferences fall
    back to opaque tokens, which may make a regex-gated branch produce nothing. The dry run reports
    it when that happens.
- **A table is treated as subject-level if it declares any static event**, and gets one row per
    subject. That is right for a patients table and approximate for anything else.
- **Nothing here validates clinical semantics.** An ETL that runs green on synthetic data can still
    be wrong about the real dataset's contents.

## Contributing

See [CONTRIBUTORS.md](CONTRIBUTORS.md). Doctests are first-class here — most of the examples in this
README and in the package docstrings run as part of the test suite.

## License

MIT. See [LICENSE](LICENSE).

# MESSY_synth

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Tests](https://github.com/mmcdermott/MESSY_synth/actions/workflows/tests.yaml/badge.svg)](https://github.com/mmcdermott/MESSY_synth/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/mmcdermott/MESSY_synth/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/mmcdermott/MESSY_synth/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/mmcdermott/MESSY_synth.svg)](https://github.com/mmcdermott/MESSY_synth/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/mmcdermott/MESSY_synth/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/mmcdermott/MESSY_synth#license)

**Generate synthetic raw data that matches the shape a [MEDS-Extract](https://github.com/mmcdermott/MEDS_extract)
v0.8 MESSY config expects — then run the whole ETL over it, with no access to the real dataset.**

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

```bash
pip install MESSY_synth # or: uv add MESSY_synth
```

MESSY_synth 0.1 reads MEDS-Extract 0.8 configs (`_table.joins`). For MEDS-Extract 0.7 configs, use
`MESSY_synth==0.0.1`.

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
MESSY-synth ./path/to/messy.yaml -o raw_synthetic/ --explain

# Run the real ETL over them.
meds-extract-run spec=./path/to/messy.yaml output_dir=meds_out/ \
	do_download=false input_dir=raw_synthetic/
```

Give the config as `./path` or an absolute path: MEDS-Extract 0.8 reads a bare `path/to/messy.yaml`
as the name of a registered pipeline. A registered name (`HIRID`) works too.

If the spec has a pre-MEDS stage that converts the raw download into the tables the config reads
(FHIR, OMOP), skip it with `'skip_stages=[<stage>]'`. MESSY_synth writes those tables directly.

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
and a name pulled in by `_table.joins` is attributed to the table it actually lives in.

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

## Explicit column generators

For representations the config cannot express, register a generator. For example, a timestamp
assembled from a date and a clock column constrains only the assembled string, so neither column's
own format is inferred:

```python
>>> from MESSY_synth import chrono_formatter
>>> split_cfg = MessyConfig.parse({
...     "etl": {"dataset_name": "SplitTime", "raw_dataset_version": "1"},
...     "visits": {
...         "_defaults": {"subject_id": "$pid"},
...         "visit": {"code": "VISIT", "time": 'f"{$date[0:10]} {$clock}"::?"%Y-%m-%d %H:%M:%S"'},
...     },
... })
>>> generators = {
...     ("visits", "date"): chrono_formatter("%Y-%m-%d"),
...     ("visits", "clock"): chrono_formatter("%H:%M:%S"),
... }
>>> with tempfile.TemporaryDirectory() as d:
...     result = synthesize(split_cfg, d, column_generators=generators)
...     result.ok
True

```

- Keys name raw `(table_prefix, column)` pairs, including full nested prefixes, not derived names.
- Each callable receives a naive synthetic `datetime` and returns one value. Registered ordinary
    columns in a table share an instant per row, so date/time fragments agree; direct birth/death
    roles retain the existing timeline handling. Callables must be deterministic for reproducibility.
- `chrono_formatter` uses Polars' Chrono formatting, the same format language as the ETL parser,
    including `%.f`, `%.3f`, `%.6f`, `%.9f`, and escaped `%%`. No Python directive translation is needed.
- Unknown columns, non-callables, metadata columns, and protected shared/subject/join keys are
    rejected rather than silently breaking relationships. Overrides do not change inferred constraints.
- Default validation evaluates the generated files through MESSY, including value expressions.
    Errors and events with no surviving rows fail validation. This is a check of the generated sample,
    **not proof of every possible callback output or complete branch coverage**. Lenient casts can still
    produce nulls in some rows. `generate()` alone and `synthesize(validate=False)` do not run this check.
- Each registration gets a verdict finding named `prefix:column`: `info` "registration matches the
    config: accepted by ..." when every event reading the column evaluates and keeps rows, or `error`
    "registration does not match the config: ..." naming the events that rejected it. A failing event only counts against a registration when the output slot that fails
    (`time`, `numeric_value`, ...) reads the column.

Use the same `column_generators` argument with `smoke_test()` for a full ETL run. From the CLI,
register Chrono formats with the repeatable `--column-format PREFIX/COL=FMT` option. The column is
the text after the last `/`, so nested prefixes work:

```bash
MESSY-synth ./messy.yaml -o raw_synthetic/ \
	--column-format visits/date=%Y-%m-%d \
	--column-format visits/clock=%H:%M:%S
```

Other generators (prefixed strings, numbers) need the Python API. They are not a general
row-context/RNG API.

CDSL registers its split date and clock columns this way, and eICU its discharge year and clock. See
their synthetic-ETL workflows.

## Smoke-testing an ETL

`smoke_test` generates, runs `meds-extract-run`, and then checks the *artifacts* — not just the
exit status. That distinction is the point: MEDS-Extract exits 0 when a lenient time cast nulls
every row of a table, and exits 0 when re-run into a populated output directory without executing a
single stage.

```python
from MESSY_synth import smoke_test


def test_etl_still_works(tmp_path):
    result = smoke_test("./src/MY_MEDS/configs/messy.yaml", tmp_path)
    assert result.ok, result.summary()
```

The CLI exits non-zero on any error finding, so it works directly in CI:

```bash
MESSY-synth ./src/MY_MEDS/configs/messy.yaml -o "$RUNNER_TEMP/raw" || exit 1
```

`smoke_test` runs every stage of the spec. For a spec with a pre-MEDS stage, run the two quick-start
commands with `skip_stages` instead.

## Validated against real ETLs

MESSY_synth 0.1.0 was checked against the MEDS-Extract 0.8 config of each ETL below, taken from the
ETL's MEDS-Extract 0.8 branch where that hadn't merged yet (November 2026). For each one,
`MESSY-synth` generated the raw files, and `meds-extract-run` built the full MEDS dataset from
them, suMEDS enrichment stage included. Each ETL runs the same check in its own synthetic-ETL CI
workflow.

| ETL                                                                      | Output from synthetic input           | Notes                                  |
| ------------------------------------------------------------------------ | ------------------------------------- | -------------------------------------- |
| [ALOTT](https://github.com/rvandewater/ALOTT_MEDS)                       | 3,474 events, 40 subjects, 445 codes  |                                        |
| [AmsterdamUMCdb](https://github.com/prockenschaub/AUMCdb_MEDS)           | 1,667 events, 39 subjects, 846 codes  |                                        |
| [CDSL](https://github.com/rvandewater/CDSL_MEDS)                         | 12,708 events, 40 subjects, 693 codes | 10 `--column-format` registrations     |
| [FHIR](https://github.com/rvandewater/FHIR_MEDS)                         | 2,341 events, 40 subjects, 639 codes  | skips the `fhir_pre_meds` stage        |
| [HiRID](https://github.com/rvandewater/HIRID_MEDS)                       | 486 events, 40 subjects, 28 codes     | join keys built from `_table.cols`     |
| [INSPIRE](https://github.com/rvandewater/INSPIRE_MEDS)                   | 1,773 events, 40 subjects, 166 codes  |                                        |
| [MC-MED](https://github.com/rvandewater/MC_MED_MEDS)                     | 2,327 events, 40 subjects, 455 codes  | several extract patterns on one column |
| [MIMIC-IV](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS) | 4,375 events, 40 subjects, 867 codes  |                                        |
| [NWICU](https://github.com/rvandewater/NWICU_MEDS)                       | 2,811 events, 40 subjects, 465 codes  |                                        |
| [OMOP](https://github.com/rvandewater/OMOP_MEDS)                         | 3,360 events, 40 subjects, 424 codes  | skips the `omop_pre_meds` stage        |
| [PICdb](https://github.com/rvandewater/PICdb_MEDS)                       | 3,741 events, 40 subjects, 553 codes  |                                        |
| [SICdb](https://github.com/rvandewater/SICdb_MEDS)                       | 1,688 events, 40 subjects, 260 codes  | numeric metadata keys                  |
| [Zigong](https://github.com/rvandewater/Zigong_MEDS)                     | 10,640 events, 40 subjects, 400 codes |                                        |

"Events" counts the rows of the MEDS data; "codes" counts the rows of `metadata/codes.parquet`. In
AmsterdamUMCdb, one of the 40 generated subjects has no events left after extraction.

[eICU](https://github.com/rvandewater/eICU_MEDS) is not in the table yet: its config is still on
MEDS-Extract 0.7. With `joins:`, a `dataset_version` under `sources:`, and two registrations for its
discharge year and clock, MESSY_synth generates data for it with no errors; the full ETL hasn't been
run.

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

# Demo: running real ETLs on synthetic data

`smoke_matrix.py` points MESSY_synth at a set of MEDS-Extract v0.7 MESSY configs, generates
synthetic sources for each, runs the real ETL over them, and prints a pass/fail matrix. No
credentials, no data-use agreements, and no real data are involved at any point.

```bash
uv run python demo/smoke_matrix.py --workdir /tmp/messy_demo \
	"MIMIC-IV=/path/to/MIMIC_IV_MEDS/src/MIMIC_IV_MEDS/configs/event_configs.yaml" \
	"NWICU=/path/to/NWICU_MEDS/src/NWICU_MEDS/configs/messy.yaml"
```

Each argument is a `label=path` pair. The script exits non-zero if any config fails, so it doubles
as a CI gate across a fleet of ETLs.

## What "pass" means

A pass is not "the command exited 0". MEDS-Extract exits 0 in several states that are not success:
a lenient time cast can null every row of a table and drop it with only a warning, a dangling join
key silently deletes rows, and re-running into a populated output directory skips every stage and
returns 0 without executing anything. So each run is checked against the artifacts:

- `data/*/*.parquet` exists and is non-empty;
- `metadata/subject_splits.parquet` exists with no empty split;
- `metadata/codes.parquet` exists and is non-empty;
- every event declared in the config produced at least one row.

## Results

Run against every v0.7 MESSY config available at the time of writing. `n_subjects` defaults to the
smallest count that safely fills the config's split fractions.

| ETL                  | Result | Output from synthetic input                          |
| -------------------- | ------ | ---------------------------------------------------- |
| MEDS-Extract example | pass   | 185 events, 16 subjects, 64 codes (40 described)     |
| MIMIC-IV             | pass   | 7,307 events, 40 subjects, 635 codes (312 described) |
| NWICU                | pass   | 2,952 events, 40 subjects, 275 codes (130 described) |
| SICdb                | pass   | 851 events, 40 subjects, 250 codes                   |
| AmsterdamUMCdb       | pass   | 5,758 events, 40 subjects, 731 codes                 |
| HiRID                | fail   | pre-existing config bug (below)                      |
| INSPIRE              | fail   | pre-existing config bug (below)                      |

SICdb and AmsterdamUMCdb report no described codes because neither config declares any `_metadata`
blocks — there is nothing to join.

eICU is absent because no v0.7 MESSY config exists for it yet; only a 0.6-era `event_configs.yaml`
does, which this tool does not read.

## The two failures are real bugs in those configs

Neither is a limitation of the generator. Both configs would fail on the real dataset too, and
MESSY_synth reports each with the specific table or column at fault.

### HiRID — one column used two incompatible ways

`raw_stage/observation_tables.datetime` is parsed as a string in its own table:

```yaml
raw_stage/observation_tables:
  observation:
    time: $datetime::?"%Y-%m-%d %H:%M:%S"
```

but the same raw column is pulled through an aggregated join into `reference_data/general_table`
and used as a timestamp with no cast:

```yaml
reference_data/general_table:
  _table:
    join:
      raw_stage/observation_tables: {key: patientid, cols: {datetime: max}}
    cols:
      date_of_death: $datetime if $_died
  discharge:
    time: $datetime
```

A raw column cannot be both text and a timestamp. `MESSY_synth` reports it statically, before the
ETL runs:

```text
ERROR    raw_stage/observation_tables.datetime: read both as a formatted string (via strptime) and
directly as a timestamp. One raw column cannot be both, so whichever site omits the cast will fail
at extraction with 'the time expression produced dtype String'. This is a bug in the config, not in
the generated data.
```

The fix is to cast at the second site too, or to do the `max` over a parsed column.

### INSPIRE — a self-join

`operations` declares `_table.join` against `operations`, then reads the polars suffix columns
`$age_right` and `$admission_time_right`. MEDS-Extract 0.7 rejects this at config-parse time:

```text
ValueError: Table 'operations' joins to itself. A self-join cannot work: every joined column
already exists on the left side (same file), and MESSY does not support suffixed join outputs.
Compute per-group reductions of a table's own columns in a pre-processing step instead.
```

The config never loads, so no data can help. It needs a pre-processing step, or a MESSY feature for
per-group reductions.

## Reproducing

The paths above are machine-specific; substitute your own checkouts. The generated data is written
under `--workdir` and can be inspected directly — every file is CSV (or parquet where a config
requires real dtypes), and `_MESSY_synth_manifest.json` records what was written.

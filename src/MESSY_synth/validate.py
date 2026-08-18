"""Check a generated dataset against the ways a MEDS-Extract run fails *quietly*.

Most of what can go wrong with synthetic source data does not raise. MEDS-Extract joins are left
joins, so a dangling key nulls the joined columns, nulls ``subject_id``, and the row is dropped
later without a word. A lenient time cast (``::?"%fmt"``) nulls anything it cannot parse and drops
the row with only a warning — an entire table can vanish while the run exits 0. A ``_metadata``
dictionary whose keys do not match the event's rendered code matches nothing and simply produces
null descriptions. In every case the pipeline reports success and the output is empty or hollow.

This module therefore does two things:

- :func:`check_plan` runs static checks against the config and plan — the failures that are
  knowable before a single row is written, such as a subject count too small for the configured
  split fractions.
- :func:`dry_run` evaluates the config's *own* expressions over the generated files and counts how
  many rows each event actually yields. An event that produces zero rows is the signal that
  matters, and it is invisible from the pipeline's own exit code.

Together they are what makes the generated data usable as a smoke test rather than merely as a
plausible-looking directory.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from .constraints import ValueKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MEDS_extract.config import MessyConfig

    from .plan import DatasetPlan

logger = logging.getLogger(__name__)

#: MEDS-Transforms' split fractions when a config declares none.
DEFAULT_SPLIT_FRACS = {"train": 0.8, "tuning": 0.1, "held_out": 0.1}

#: Expected subject count below which the rarest split can round away to nothing.
#: ``split_and_shard_subjects`` permutes the split names before rounding, so a cohort near this
#: boundary fails for some seeds and passes for others; below it, failure is the common case.
MIN_RAREST_SPLIT_MASS = 0.5


@dataclass(frozen=True)
class Finding:
    """One validation result.

    Attributes:
        level: ``"error"``, ``"warning"``, or ``"info"``.
        where: What the finding is about — a table prefix, a column, or ``"config"``.
        message: Human-readable explanation, including the remedy where there is one.
    """

    level: str
    where: str
    message: str

    def __str__(self) -> str:
        """Render as a single aligned log line.

        Returns:
            The rendered line.

        Examples:
            >>> print(Finding("warning", "labs", "no rows"))
            WARNING  labs: no rows
        """
        return f"{self.level.upper():<8} {self.where}: {self.message}"


def split_fracs(cfg: MessyConfig) -> dict[str, float]:
    """Return the split fractions this config will run with.

    Args:
        cfg: The parsed MESSY config.

    Returns:
        The declared ``split_fracs``, or MEDS-Transforms' defaults.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> split_fracs(MessyConfig.parse({"t": {"e": {"code": "X", "time": None}}}))
        {'train': 0.8, 'tuning': 0.1, 'held_out': 0.1}
        >>> split_fracs(MessyConfig.parse({
        ...     "etl": {"split_fracs": {"train": 0.5, "held_out": 0.5}},
        ...     "t": {"e": {"code": "X", "time": None}},
        ... }))
        {'train': 0.5, 'held_out': 0.5}
    """
    declared = (cfg.etl.stage_options or {}).get("split_fracs")
    return dict(declared) if declared else dict(DEFAULT_SPLIT_FRACS)


def recommended_n_subjects(cfg: MessyConfig) -> int:
    """Return the smallest subject count that safely populates every split.

    ``split_and_shard_subjects`` permutes the split names before rounding, so a cohort small
    enough that the rarest split rounds to zero fails for *some* seeds and passes for others —
    a flaky smoke test. Requiring at least four subjects in the rarest split removes the
    seed-dependence entirely.

    Args:
        cfg: The parsed MESSY config.

    Returns:
        The recommended subject count, never below 10.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> bare = MessyConfig.parse({"t": {"e": {"code": "X", "time": None}}})
        >>> recommended_n_subjects(bare)
        40
        >>> even = MessyConfig.parse({
        ...     "etl": {"split_fracs": {"train": 0.5, "tuning": 0.25, "held_out": 0.25}},
        ...     "t": {"e": {"code": "X", "time": None}},
        ... })
        >>> recommended_n_subjects(even)
        16
    """
    fracs = [f for f in split_fracs(cfg).values() if isinstance(f, int | float) and f > 0]
    if not fracs:
        return 10
    return max(10, math.ceil(4 / min(fracs)))


def check_plan(cfg: MessyConfig, plan: DatasetPlan, fmt: str = "csv") -> list[Finding]:
    """Run the static checks that do not require generated data.

    Args:
        cfg: The parsed MESSY config.
        plan: The generation plan.
        fmt: The concrete output format that will be written.

    Returns:
        The findings, most severe first.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> from MESSY_synth.plan import build_plan
        >>> cfg = MessyConfig.parse({
        ...     "etl": {"dataset_name": "D", "raw_dataset_version": "1"},
        ...     "t": {"_defaults": {"subject_id": "$pid"},
        ...           "e": {"code": "X", "time": '$ts::"%Y-%m-%d"'}},
        ... })
        >>> check_plan(cfg, build_plan(cfg, n_subjects=40))
        []

        A cohort so small that the rarest split rounds away to nothing is an error — the run then
        fails for some seeds and passes for others, which is worse than failing outright:

        >>> for f in check_plan(cfg, build_plan(cfg, n_subjects=3)):
        ...     print(f)
        ERROR    config: 3 subjects is too few for split_fracs {'train': 0.8, 'tuning': 0.1,
        'held_out': 0.1}: the rarest split gets 0.30 subjects and will round to zero. Use at
        least 40.

        A cohort that will populate every split but leaves them thin is only advisory, so a
        deliberately small run is not blocked:

        >>> for f in check_plan(cfg, build_plan(cfg, n_subjects=8)):
        ...     print(f)
        WARNING  config: 8 subjects leaves the rarest split with about 0.8 subjects. It will
        populate, but 40 would give every split a usable size.

        Writing a bare timestamp column to CSV is flagged, because polars' CSV inference will
        deliver it as a string and the run will fail at extraction:

        >>> bare = MessyConfig.parse({
        ...     "etl": {"dataset_name": "D", "raw_dataset_version": "1"},
        ...     "t": {"_defaults": {"subject_id": "$pid"}, "e": {"code": "X", "time": "$ts"}},
        ... })
        >>> for f in check_plan(bare, build_plan(bare, n_subjects=40), fmt="csv"):
        ...     print(f)
        WARNING  t.ts: used as a timestamp with no strptime format, but the output format is
        'csv', which cannot carry a datetime dtype. Use format='parquet'.

        A column read both as a formatted string and as a native timestamp is a contradiction the
        config cannot satisfy with any input, and is reported as such:

        >>> conflicted = MessyConfig.parse({
        ...     "etl": {"dataset_name": "D", "raw_dataset_version": "1"},
        ...     "obs": {"_defaults": {"subject_id": "$pid"},
        ...             "e": {"code": "X", "time": '$ts::"%Y-%m-%d"'}},
        ...     "gen": {"_defaults": {"subject_id": "$pid"},
        ...             "_table": {"join": {"obs": {"key": "pid", "cols": {"ts": "max"}}}},
        ...             "d": {"code": "Y", "time": "$ts"}},
        ... })
        >>> print(check_plan(conflicted, build_plan(conflicted, n_subjects=40))[0])
        ERROR    obs.ts: read both as a formatted string (via strptime) and directly as a
        timestamp. One raw column cannot be both, so whichever site omits the cast will fail at
        extraction with 'the time expression produced dtype String'. This is a bug in the config,
        not in the generated data.

        A config that omits what ``meds-extract-run`` requires is reported before any data is
        written, since the run would refuse to start:

        >>> incomplete = MessyConfig.parse({"t": {"e": {"code": "X", "time": None}}})
        >>> [f.message for f in check_plan(incomplete, build_plan(incomplete, n_subjects=40))]
        ['No raw dataset version declared...add `dataset_version` to `sources:` or
        `raw_dataset_version` to `etl:`...']
    """
    findings: list[Finding] = []

    fracs = [f for f in split_fracs(cfg).values() if isinstance(f, int | float) and f > 0]
    rarest = min(fracs) if fracs else 1.0
    recommended = recommended_n_subjects(cfg)
    if plan.n_subjects * rarest < MIN_RAREST_SPLIT_MASS:
        findings.append(
            Finding(
                "error",
                "config",
                f"{plan.n_subjects} subjects is too few for split_fracs {split_fracs(cfg)}: the "
                f"rarest split gets {plan.n_subjects * rarest:.2f} subjects and will round to zero. "
                f"Use at least {recommended}.",
            )
        )
    elif plan.n_subjects < recommended:
        findings.append(
            Finding(
                "warning",
                "config",
                f"{plan.n_subjects} subjects leaves the rarest split with about "
                f"{plan.n_subjects * rarest:.1f} subjects. It will populate, but "
                f"{recommended} would give every split a usable size.",
            )
        )

    if cfg.sources_version is None and cfg.etl.raw_dataset_version is None:
        findings.append(
            Finding(
                "error",
                "config",
                "No raw dataset version declared, so `meds-extract-run` will refuse to start: "
                "add `dataset_version` to `sources:` or `raw_dataset_version` to `etl:`.",
            )
        )

    for table in plan.tables:
        for column in table.columns:
            observed = column.constraint.observed_kinds
            if ValueKind.DATETIME_STR in observed and ValueKind.DATETIME in observed:
                findings.append(
                    Finding(
                        "error",
                        f"{table.prefix}.{column.name}",
                        "read both as a formatted string (via strptime) and directly as a "
                        "timestamp. One raw column cannot be both, so whichever site omits the "
                        "cast will fail at extraction with 'the time expression produced dtype "
                        "String'. This is a bug in the config, not in the generated data.",
                    )
                )

    if fmt.startswith("csv"):
        for table in plan.tables:
            for column in table.columns:
                kind = column.constraint.kind
                if kind in (ValueKind.DATETIME, ValueKind.DATE) and not column.constraint.datetime_formats:
                    findings.append(
                        Finding(
                            "warning",
                            f"{table.prefix}.{column.name}",
                            f"used as a timestamp with no strptime format, but the output format "
                            f"is {fmt!r}, which cannot carry a datetime dtype. Use "
                            f"format='parquet'.",
                        )
                    )

    order = {"error": 0, "warning": 1, "info": 2}
    return sorted(findings, key=lambda f: (order.get(f.level, 3), f.where))


def dry_run(cfg: MessyConfig, input_dir: str | Path) -> pl.DataFrame:
    """Evaluate the config's own expressions over written data and count what each event yields.

    This is the check that catches silent emptiness. It reproduces what
    ``convert_to_MEDS_events`` does — scan the table, apply the join, materialize ``subject_id``
    and ``_table.cols``, then evaluate each event's ``code`` and ``time`` — and reports how many
    rows survive with a usable code.

    Args:
        cfg: The parsed MESSY config.
        input_dir: Directory holding the generated source files.

    ``rows_out`` counts *source rows that would yield an event*, which is what matters for
    catching silent emptiness. It is an upper bound on the events finally written: extraction ends
    with ``.unique()``, so rows producing byte-identical events collapse into one.

    Returns:
        One row per event, with ``table``, ``event``, ``rows_in``, ``rows_out``, and ``error``.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> from MESSY_synth.generate import generate, GenerationOptions
        >>> from MESSY_synth.writer import write_dataset
        >>> cfg = MessyConfig.parse({
        ...     "etl": {"dataset_name": "D", "raw_dataset_version": "1"},
        ...     "labs": {
        ...         "_defaults": {"subject_id": "$pid"},
        ...         "lab": {"code": 'f"LAB//{$itemid}"', "time": '$ts::"%Y-%m-%d %H:%M:%S"'},
        ...     },
        ... })
        >>> ds = generate(cfg, GenerationOptions(seed=0, n_subjects=10, rows_per_subject=2))
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = write_dataset(ds, d, write_manifest=False)
        ...     dry_run(cfg, d)
        shape: (1, 5)
        ┌───────┬───────┬─────────┬──────────┬───────┐
        │ table ┆ event ┆ rows_in ┆ rows_out ┆ error │
        │ ---   ┆ ---   ┆ ---     ┆ ---      ┆ ---   │
        │ str   ┆ str   ┆ i64     ┆ i64      ┆ str   │
        ╞═══════╪═══════╪═════════╪══════════╪═══════╡
        │ labs  ┆ lab   ┆ 20      ┆ 18       ┆ null  │
        └───────┴───────┴─────────┴──────────┴───────┘

        Two rows of twenty are lost to the deliberate nulls injected into ``itemid``: under 0.7 a
        null component null-propagates through the interpolated code and the row is dropped. That
        path is worth exercising, which is why nulls are generated at all.
    """
    input_dir = Path(input_dir)
    rows: list[dict] = []
    for table in cfg.event_tables:
        try:
            frame = table.prepare(table.scan(input_dir)).collect()
            rows_in = frame.height
        except Exception as e:
            rows.append(
                {
                    "table": table.input_prefix,
                    "event": "*",
                    "rows_in": 0,
                    "rows_out": 0,
                    "error": f"{type(e).__name__}: {e}"[:200],
                }
            )
            continue

        for event in table.events:
            record = {
                "table": table.input_prefix,
                "event": event.name,
                "rows_in": rows_in,
                "rows_out": 0,
                "error": None,
            }
            try:
                exprs = event.polars_exprs
                # `with_columns`, not `select`: an event whose code and time are both literals
                # (`code: MEDS_BIRTH, time: null`) selects to a single broadcast row, which would
                # report one event where the ETL emits one per source row.
                out = frame.with_columns(
                    _code=exprs["code"],
                    _time=exprs.get("time", pl.lit(None, dtype=pl.Datetime)),
                ).select(code=pl.col("_code"), time=pl.col("_time"))
                # A null code drops the row outright. A null *time* also drops it, but only for a
                # timed event — a static event (`time: null`) is null-timed by definition. Counting
                # code alone would overstate the yield of, say, a `MEDS_DEATH` event whose
                # timestamp is null for every surviving subject.
                keep = pl.col("code").is_not_null()
                if not event.is_static:
                    # Reproduce `EventConfig.extract`'s dtype guard. A timed event whose `time`
                    # expression is not temporally typed aborts the real run, but evaluating the
                    # expression here succeeds — so without this check the dry run reports a healthy
                    # row count for a config that cannot run at all. HiRID is exactly this case: one
                    # raw column read as a string in one table and as a timestamp in another.
                    time_dtype = out.schema["time"]
                    if time_dtype != pl.Null and not isinstance(time_dtype, pl.Datetime | pl.Date):
                        record["error"] = (
                            f"the `time` expression produced dtype {time_dtype}, not a "
                            f"date/datetime. MEDS-Extract rejects this at extraction."
                        )
                        rows.append(record)
                        continue
                    keep = keep & pl.col("time").is_not_null()
                record["rows_out"] = int(out.select(keep.sum()).item())
            except Exception as e:
                record["error"] = f"{type(e).__name__}: {e}"[:200]
            rows.append(record)

    schema = {
        "table": pl.String,
        "event": pl.String,
        "rows_in": pl.Int64,
        "rows_out": pl.Int64,
        "error": pl.String,
    }
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def findings_from_dry_run(result: pl.DataFrame) -> list[Finding]:
    """Convert a dry-run table into findings.

    Args:
        result: The frame returned by :func:`dry_run`.

    Returns:
        One finding per event that errored or produced nothing.

    Examples:
        >>> res = pl.DataFrame({
        ...     "table": ["labs", "vitals"], "event": ["lab", "vital"],
        ...     "rows_in": [10, 10], "rows_out": [10, 0], "error": [None, None],
        ... })
        >>> for f in findings_from_dry_run(res):
        ...     print(f)
        ERROR    vitals/vital: produced 0 events from 10 source rows. The extracted dataset will be
        missing this event entirely.
    """
    findings: list[Finding] = []
    for row in result.iter_rows(named=True):
        where = f"{row['table']}/{row['event']}"
        if row["error"]:
            findings.append(Finding("error", where, f"failed to evaluate: {row['error']}"))
        elif row["rows_out"] == 0:
            findings.append(
                Finding(
                    "error",
                    where,
                    f"produced 0 events from {row['rows_in']} source rows. The extracted dataset "
                    f"will be missing this event entirely.",
                )
            )
    return findings

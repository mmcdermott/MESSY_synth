"""Materialize a :class:`~MESSY_synth.plan.DatasetPlan` into in-memory polars frames.

This is where the plan's abstract promises become rows. The three that matter, and how they are
kept:

- **One subject universe.** Every column in the ``__subjects__`` pool is filled from the same list
  of integer ids, so events extracted from different files land on the same subjects.
- **Joins that land.** A column marked ``covers_pool`` *enumerates* its pool positionally rather
  than sampling it, guaranteeing every referenced key exists on the far side. Everything else
  samples, so the data still looks like a many-to-one relationship rather than a lockstep zip.
- **Coherent time.** Each subject gets a :class:`~MESSY_synth.values.SubjectTimeline` up front, and
  every timestamp for that subject is drawn from inside it — births before the window, deaths
  after, ordinary events within.

Generation is deterministic: each column seeds its own RNG from ``(seed, prefix, column)``, so the
same config and seed always produce byte-identical output, and adding a table to a config does not
perturb the values in unrelated tables.
"""

from __future__ import annotations

import itertools
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import polars as pl

from .constraints import AssembledTimestamp, ColumnConstraint, ValueKind
from .plan import (
    DEFAULT_ROWS_PER_SUBJECT,
    DEFAULT_VOCAB_SIZE,
    MAX_METADATA_ROWS,
    SUBJECT_KEY_POOL,
    SUBJECT_POOL,
    build_plan,
)
from .values import ValueFactory, build_timeline, categorical_pool, format_datetime

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MEDS_extract.config import MessyConfig

    from .plan import ColumnPlan, DatasetPlan, TablePlan
    from .values import SubjectTimeline

logger = logging.getLogger(__name__)

#: Share of rows in a numeric column that take one of the config's compared-against literals.
REQUIRED_VALUE_SHARE = 0.35

#: Draw attempts per wanted member when filling a numeric pool with distinct values.
NUMERIC_POOL_DRAW_FACTOR = 20

#: User-supplied rendering for a source column, given a coherent synthetic timestamp.
ColumnGenerator = Callable[[datetime], object]


@dataclass(frozen=True)
class GenerationOptions:
    """Knobs controlling the size and shape of the generated dataset.

    Attributes:
        seed: Master seed. Output is a pure function of (config, options), so a given seed
            reproduces a dataset exactly.
        n_subjects: How many distinct subjects to invent.
        rows_per_subject: Rows per subject in each timed event table.
        vocab_size: Distinct values in each categorical vocabulary.
        null_fraction: Fraction of nullable values emitted as nulls, so the config's ``??``
            coalescing and null-drop paths are actually exercised.
        death_probability: Fraction of subjects given a death timestamp.
        numeric_ranges: Per-column-name numeric bounds overriding the default ``(1, 100)``.
        column_generators: Explicit renderers keyed by raw ``(prefix, column)`` pairs. Each receives
            a synthetic timestamp; registered ordinary columns share one instant per row and direct
            birth/death roles use their timeline bounds. Callables must be deterministic to preserve
            reproducibility. Metadata and protected shared/subject/join keys cannot be overridden.
            This does not alter inferred constraints; use ``synthesize`` for MESSY validation.
    """

    seed: int = 0
    n_subjects: int = 20
    rows_per_subject: int = DEFAULT_ROWS_PER_SUBJECT
    vocab_size: int = DEFAULT_VOCAB_SIZE
    null_fraction: float = 0.05
    death_probability: float = 0.3
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    column_generators: dict[tuple[str, str], ColumnGenerator] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedDataset:
    """The generated frames plus the plan that produced them.

    Attributes:
        plan: The plan this dataset was generated from, retained so callers can explain or verify
            the output without recomputing it.
        frames: One frame per source file, keyed by table prefix.
        timelines: The invented per-subject timelines, keyed by subject id.
        truncated_metadata: Metadata prefixes whose key combinations exceeded
            :data:`~MESSY_synth.plan.MAX_METADATA_ROWS`, mapped to how many were dropped. Codes
            built from a dropped combination have no dictionary row, so this is surfaced as a
            finding rather than left to a log line.
        assembly_conflicts: Assembled timestamps (see
            :class:`~MESSY_synth.constraints.AssembledTimestamp`) whose pieces could not be cut to fit
            in some rows, because another constraint fixed one of their columns, mapped to how many
            rows failed. For a ``coalesce``, the key is its first branch.
    """

    plan: DatasetPlan
    frames: dict[str, pl.DataFrame]
    timelines: dict[int, SubjectTimeline]
    truncated_metadata: dict[str, int] = field(default_factory=dict)
    assembly_conflicts: dict[AssembledTimestamp, int] = field(default_factory=dict)

    def summary(self) -> pl.DataFrame:
        """Return a one-row-per-file overview of what was generated.

        Returns:
            A frame with ``prefix``, ``rows``, ``columns``, and ``kind`` columns.
        """
        return pl.DataFrame(
            {
                "prefix": [t.prefix for t in self.plan.tables],
                "rows": [self.frames[t.prefix].height for t in self.plan.tables],
                "columns": [self.frames[t.prefix].width for t in self.plan.tables],
                "kind": [t.kind for t in self.plan.tables],
            }
        )


def generate(cfg: MessyConfig, options: GenerationOptions | None = None) -> GeneratedDataset:
    """Generate a complete synthetic source dataset for one MESSY config.

    Args:
        cfg: The parsed MESSY config.
        options: Size and shape knobs; defaults to :class:`GenerationOptions`.

    Returns:
        The :class:`GeneratedDataset`.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> cfg = MessyConfig.parse({
        ...     "etl": {"dataset_name": "Demo"},
        ...     "_defaults": {"subject_id": "$patient_id"},
        ...     "patients": {
        ...         "dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'},
        ...         "sex": {"code": 'f"SEX//{$sex}"', "time": None},
        ...     },
        ...     "labs": {
        ...         "lab": {
        ...             "code": 'f"LAB//{$itemid}"',
        ...             "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
        ...             "numeric_value": "$value",
        ...             "_metadata": {"d_labitems": {"itemid": "$itemid", "description": "$label"}},
        ...         },
        ...     },
        ... })
        >>> ds = generate(cfg, GenerationOptions(seed=0, n_subjects=4, rows_per_subject=2))
        >>> ds.summary()
        shape: (3, 4)
        ┌────────────┬──────┬─────────┬───────────────┐
        │ prefix     ┆ rows ┆ columns ┆ kind          │
        │ ---        ┆ ---  ┆ ---     ┆ ---           │
        │ str        ┆ i64  ┆ i64     ┆ str           │
        ╞════════════╪══════╪═════════╪═══════════════╡
        │ d_labitems ┆ 8    ┆ 2       ┆ metadata      │
        │ labs       ┆ 8    ┆ 4       ┆ events        │
        │ patients   ┆ 4    ┆ 3       ┆ subject-level │
        └────────────┴──────┴─────────┴───────────────┘

        ``patients`` carries a static event, so it is one row per subject; the birth column is
        rendered in exactly the format the config parses it with, and the occasional null in a
        code component is deliberate — in 0.7 a null component drops the row, and that path
        deserves to be exercised too:

        >>> ds.frames["patients"].sort("patient_id")
        shape: (4, 3)
        ┌────────────┬────────────┬───────────────┐
        │ patient_id ┆ dob        ┆ sex           │
        │ ---        ┆ ---        ┆ ---           │
        │ i64        ┆ str        ┆ str           │
        ╞════════════╪════════════╪═══════════════╡
        │ 1          ┆ 1969-09-28 ┆ null          │
        │ 2          ┆ 1959-10-06 ┆ SYNTH_SEX_004 │
        │ 3          ┆ 1940-10-26 ┆ SYNTH_SEX_001 │
        │ 4          ┆ 1951-11-27 ┆ SYNTH_SEX_003 │
        └────────────┴────────────┴───────────────┘

        Every value is visibly synthetic; nothing here resembles a real vocabulary.

        The metadata dictionary covers the event table's vocabulary exactly, which is what makes
        the code-description join land instead of silently matching nothing:

        >>> set(ds.frames["labs"]["itemid"]) <= set(ds.frames["d_labitems"]["itemid"])
        True

        Generation is deterministic given a seed:

        >>> other = generate(cfg, GenerationOptions(seed=0, n_subjects=4, rows_per_subject=2))
        >>> other.frames["labs"].equals(ds.frames["labs"])
        True
    """
    options = options or GenerationOptions()
    plan = build_plan(
        cfg,
        n_subjects=options.n_subjects,
        rows_per_subject=options.rows_per_subject,
        vocab_size=options.vocab_size,
    )
    _validate_column_generators(plan, options.column_generators)

    pools = _materialize_pools(plan, options)
    subject_ids: list[int] = pools[SUBJECT_POOL]
    timelines = {
        sid: build_timeline(sid, random.Random(f"{options.seed}:timeline:{sid}"), options.death_probability)
        for sid in subject_ids
    }

    joins = {t.input_prefix: t.join for t in cfg.event_tables if t.join is not None}
    frames: dict[str, pl.DataFrame] = {}
    truncated: dict[str, int] = {}
    conflicts: dict[AssembledTimestamp, int] = {}
    for table in _generation_order(plan, joins):
        frame = _generate_table(table, plan, pools, timelines, subject_ids, options, truncated, conflicts)
        join = joins.get(table.prefix)
        if join is not None and join.input_prefix in frames:
            frame = _adopt_join_keys(frame, table, join, frames[join.input_prefix], options)
        frames[table.prefix] = frame
    return GeneratedDataset(
        plan=plan,
        frames=frames,
        timelines=timelines,
        truncated_metadata=truncated,
        assembly_conflicts=conflicts,
    )


def _validate_column_generators(
    plan: DatasetPlan, generators: dict[tuple[str, str], ColumnGenerator]
) -> None:
    """Reject generator registrations that would violate a planned source relationship."""
    source_columns = {
        (table.prefix, column.name): column
        for table in plan.tables
        if not table.is_metadata
        for column in table.columns
    }
    metadata_columns = {
        (table.prefix, column.name) for table in plan.tables if table.is_metadata for column in table.columns
    }
    for key, generator in generators.items():
        if key in metadata_columns:
            raise ValueError(
                f"column generator {key!r} targets metadata, which is generated from shared keys"
            )
        column = source_columns.get(key)
        if column is None:
            raise ValueError(f"column generator names an unknown source column {key!r}")
        if not callable(generator):
            raise TypeError(f"column generator for {key!r} is not callable")
        if _shares_a_relationship(column, plan):
            raise ValueError(f"column generator {key!r} would break a shared source relationship")


def _shares_a_relationship(column: ColumnPlan, plan: DatasetPlan) -> bool:
    """Return whether a column's values are tied to another column, so must not be overwritten."""
    pool = plan.pools.get(column.pool_id) if column.pool_id else None
    return (
        column.covers_pool
        or column.unique_in_table
        or column.pool_id in (SUBJECT_POOL, SUBJECT_KEY_POOL)
        or (pool is not None and len(pool.members) > 1)
    )


def _generation_order(plan: DatasetPlan, joins: dict) -> list[TablePlan]:
    """Order tables so every join target is built before the table that joins to it.

    Join keys are not invented independently on the two sides — the referencing table copies real
    key tuples out of the target's finished frame — so the target has to exist first.

    Args:
        plan: The dataset plan.
        joins: Table prefix to its :class:`~MEDS_extract.config.JoinConfig`.

    Returns:
        The table plans in dependency order. A cycle (which MEDS-Extract rejects for self-joins,
        but could in principle span tables) degrades to the plan's own order rather than hanging.
    """
    by_prefix = {t.prefix: t for t in plan.tables}
    ordered: list[TablePlan] = []
    placed: set[str] = set()

    def visit(prefix: str, seen: frozenset[str]) -> None:
        if prefix in placed or prefix in seen or prefix not in by_prefix:
            return
        join = joins.get(prefix)
        if join is not None:
            visit(join.input_prefix, seen | {prefix})
        if prefix not in placed:
            placed.add(prefix)
            ordered.append(by_prefix[prefix])

    for table in plan.tables:
        visit(table.prefix, frozenset())
    return ordered


def _adopt_join_keys(
    frame: pl.DataFrame,
    table: TablePlan,
    join,
    target: pl.DataFrame,
    options: GenerationOptions,
) -> pl.DataFrame:
    """Replace a table's join-key columns with real key tuples drawn from the target's frame.

    Generating the two sides independently is not good enough for a composite key: each column
    might individually come from the right pool while the *combination* exists nowhere on the far
    side, so almost every row dangles and is silently dropped. Copying whole tuples from the target
    makes every left key resolve by construction, for composite and single keys alike.

    The subject column is exempt. It already agrees across tables by drawing from the shared subject
    universe, and overwriting it would break the round-robin assignment that gives every subject a
    row — and, in a subject-level table, would put two rows on one subject.

    Args:
        frame: The freshly generated frame.
        table: Its plan.
        join: The table's join config.
        target: The already-generated frame of the join target.
        options: The generation options, for the seed.

    Returns:
        The frame with its join keys adopted from the target.
    """
    pairs = [
        (left, right)
        for left, right in zip(join.left_on, join.right_on, strict=True)
        if left != table.subject_column and left in frame.columns and right in target.columns
    ]
    if not pairs or target.height == 0:
        return frame

    rng = random.Random(f"{options.seed}:joinkeys:{table.prefix}")
    picks = [rng.randrange(target.height) for _ in range(frame.height)]
    return frame.with_columns(
        *(target[right].gather(picks).alias(left) for left, right in pairs),
    )


def _materialize_pools(plan: DatasetPlan, options: GenerationOptions) -> dict[str, list]:
    """Turn each planned pool into its concrete list of values.

    Args:
        plan: The dataset plan.
        options: The generation options.

    Returns:
        A mapping of pool id to values.
    """
    pools: dict[str, list] = {SUBJECT_POOL: list(range(1, options.n_subjects + 1))}
    for pool_id, pool in plan.pools.items():
        if pool_id == SUBJECT_POOL:
            continue
        # Name the pool after its first member column so tokens are self-describing.
        column = pool.members[0][1] if pool.members else pool_id
        rng = random.Random(f"{options.seed}:pool:{pool_id}")
        if pool.kind in (ValueKind.NUMERIC, ValueKind.INTEGER):
            required = list(
                dict.fromkeys(
                    value
                    for literal in pool.constraint.required_values
                    if (value := _as_number(literal, pool.kind is ValueKind.INTEGER)) is not None
                )
            )
            factory = ValueFactory(rng, numeric_ranges=options.numeric_ranges)
            # Filler must be distinct: a repeated member becomes a duplicate dictionary key. A
            # narrow range may hold fewer distinct numbers than the pool size; stop rather than
            # loop forever, since every consumer indexes by the pool's actual length.
            members = dict.fromkeys(required)
            target = len(required) + max(1, pool.size - len(required))
            for _ in range(target * NUMERIC_POOL_DRAW_FACTOR):
                if len(members) >= target:
                    break
                members.setdefault(
                    factory.numeric(
                        column,
                        integral=pool.kind is ValueKind.INTEGER,
                        inferred_range=pool.constraint.numeric_range,
                    )
                )
            pools[pool_id] = list(members)
        else:
            pools[pool_id] = categorical_pool(column, pool.constraint, pool.size, rng)
    return pools


def _generate_table(
    table: TablePlan,
    plan: DatasetPlan,
    pools: dict[str, list],
    timelines: dict[int, SubjectTimeline],
    subject_ids: list[int],
    options: GenerationOptions,
    truncated: dict[str, int],
    conflicts: dict[AssembledTimestamp, int],
) -> pl.DataFrame:
    """Generate one file's frame.

    Args:
        table: The table plan.
        plan: The dataset plan (for pool metadata).
        pools: Materialized pool values.
        timelines: Per-subject timelines.
        subject_ids: The subject universe.
        options: The generation options.
        truncated: Accumulator recording metadata tables whose key product hit the row cap.
        conflicts: Accumulator recording assembled timestamps that could not be cut to fit.

    Returns:
        The generated frame.
    """
    if table.is_metadata:
        return _generate_metadata_table(table, pools, options, truncated)

    # The plan sizes tables from each pool's *requested* size, but a materialized pool can end up
    # longer — required values, coalesce fallbacks and regex-derived members are all added on top.
    # A covering column must enumerate whatever the pool actually holds, so the row count is raised
    # to match; otherwise the tail of the pool never appears and every row referencing it dangles.
    n_rows = max(
        [table.n_rows, *(len(pools[c.pool_id]) for c in table.columns if c.covers_pool and c.pool_id)]
    )
    # Round-robin rather than random assignment, so every subject is represented in every table
    # instead of a few subjects hogging the rows by chance.
    row_subjects = [subject_ids[i % len(subject_ids)] for i in range(n_rows)] if subject_ids else []

    data: dict[str, list] = {}
    for column in table.columns:
        data[column.name] = _generate_column(
            column, table, plan, pools, timelines, row_subjects, n_rows, options
        )
    assemblies = [a for a in plan.assemblies if a.prefix == table.prefix]
    if assemblies:
        _apply_assemblies(assemblies, table, plan, data, timelines, row_subjects, n_rows, options, conflicts)
    return pl.DataFrame(data, strict=False)


def _apply_assemblies(
    assemblies: list[AssembledTimestamp],
    table: TablePlan,
    plan: DatasetPlan,
    data: dict[str, list],
    timelines: dict[int, SubjectTimeline],
    row_subjects: list[int],
    n_rows: int,
    options: GenerationOptions,
    conflicts: dict[AssembledTimestamp, int],
) -> None:
    """Overwrite the pieces of assembled timestamps so each assembly parses under its format.

    Each row gets one instant, rendered whole with the assembly's format and cut at the template's
    literals. No column is given a format of its own: any cut that respects the literals reassembles
    into the rendered string. A column is left as generated (and the cut must fit around it) when
    something else owns its values — a user registration, a shared key, a birth/death role, or its
    own ``strptime`` format. Values already null stay null, preserving null injection.

    Args:
        assemblies: This table's assembled timestamps.
        table: The table plan.
        plan: The dataset plan.
        data: The generated columns, modified in place.
        timelines: Per-subject timelines.
        row_subjects: The subject id assigned to each row.
        n_rows: The number of rows.
        options: The generation options.
        conflicts: Accumulator for rows that could not be cut to fit.
    """
    columns = {column.name: column for column in table.columns}
    free = {
        name
        for name, column in columns.items()
        if (table.prefix, name) not in options.column_generators
        and not _shares_a_relationship(column, plan)
        and column.constraint.effective_temporal_role is None
        and column.constraint.kind not in (ValueKind.DATETIME, ValueKind.DATE, ValueKind.DATETIME_STR)
    }
    # The same seed as registered columns, so registered fragments and cut pieces share an instant.
    rng = random.Random(f"{options.seed}:{table.prefix}:registered-time")
    instants = pl.Series(_row_instants(timelines, row_subjects, n_rows, rng), dtype=pl.Datetime("us"))

    # Branches of one coalesce are alternatives; every other assembly must parse on its own.
    groups: dict[object, list[AssembledTimestamp]] = {}
    for assembly in assemblies:
        groups.setdefault(assembly.group or assembly, []).append(assembly)

    assigned: list[set[str]] = [set() for _ in range(n_rows)]
    for alternatives in groups.values():
        rendered = []
        for assembly in alternatives:
            try:
                rendered.append(instants.dt.strftime(assembly.fmt).to_list())
            except Exception:  # an invalid format; the dry run reports the expression itself
                rendered.append([None] * n_rows)
        failed = 0
        for i in range(n_rows):
            for assembly, strings in zip(alternatives, rendered, strict=True):
                current = {c: data[c][i] for c in assembly.columns}
                nulls = {c for c, v in current.items() if v is None}
                fixed = {
                    c: str(v)
                    for c, v in current.items()
                    if c not in nulls and (c not in free or c in assigned[i])
                }
                cut = (
                    None
                    if strings[i] is None
                    else _cut(strings[i], assembly.literals, assembly.pieces, fixed)
                )
                if cut is None and nulls:
                    break  # the assembled value is null in this row anyway
                if cut is not None:
                    # Injected nulls stay null; the other pieces still get realistic values.
                    for column, value in cut.items():
                        if column not in nulls:
                            data[column][i] = value
                            assigned[i].add(column)
                    break
            else:
                failed += 1
        if failed:
            conflicts[alternatives[0]] = conflicts.get(alternatives[0], 0) + failed


def _row_instants(
    timelines: dict[int, SubjectTimeline], row_subjects: list[int], n_rows: int, rng: random.Random
) -> list[datetime]:
    """Draw one ordinary event instant per row, as :func:`_generate_temporal` does."""
    out = []
    for i in range(n_rows):
        subject = row_subjects[i] if i < len(row_subjects) else None
        timeline = timelines.get(subject) if subject is not None else None
        if timeline is None:
            timeline = build_timeline(0, random.Random(f"anon:{i}"))
        out.append(timeline.sample(rng))
    return out


#: Filler for the characters of a column outside every slice the config reads.
ASSEMBLY_PAD = "_"

#: A column value being cut: its known characters by position, and its length if pinned.
_Span = tuple[dict[int, str], int | None]


def _cut(
    rendered: str,
    literals: tuple[str, ...],
    pieces: tuple[tuple[str, int, int | None], ...],
    fixed: dict[str, str],
) -> dict[str, str] | None:
    """Cut a rendered timestamp into column values that reassemble into it.

    Args:
        rendered: The whole timestamp, rendered with the assembly's format.
        literals: The template text around the pieces.
        pieces: ``(column, start, stop)`` per piece.
        fixed: Values that must be kept as they are.

    Returns:
        The new value of each non-fixed column, or None if no cut fits.

    Examples:
        >>> _cut("2015-06-01 13:45:00", ("", " ", ""), (("date", 0, 10), ("clock", 0, None)), {})
        {'date': '2015-06-01', 'clock': '13:45:00'}

        Separators that also occur inside the timestamp are not a problem:

        >>> _cut("2015-06-01", ("", "-", "-", ""), (("y", 0, None), ("m", 0, None), ("d", 0, None)), {})
        {'y': '2015', 'm': '06', 'd': '01'}

        Characters outside the slice are padding, and a fixed value is fitted around:

        >>> _cut("2015-06-01", ("", ""), (("raw", 3, 13),), {})
        {'raw': '___2015-06-01'}
        >>> _cut("2015-06-01 13:45", ("", " ", ""), (("d", 0, None), ("t", 0, None)), {"d": "2015-06-01"})
        {'t': '13:45'}
        >>> _cut("2015-06-01 13:45", ("", " ", ""), (("d", 0, None), ("t", 0, None)), {"d": "nope"}) is None
        True

        Two slices of one column must agree on a single value:

        >>> _cut("2015/06", ("", "/", ""), (("s", 0, 4), ("s", 5, 7)), {})
        {'s': '2015_06'}
    """

    def place(span: _Span | None, start: int, stop: int | None, piece: str) -> _Span | None:
        chars, end = span if span is not None else ({}, None)
        new_end = start + len(piece) if stop is None else None
        if end is not None and new_end is not None and end != new_end:
            return None
        end = end if new_end is None else new_end
        merged = dict(chars)
        for offset, char in enumerate(piece):
            if merged.setdefault(start + offset, char) != char:
                return None
        if end is not None and any(position >= end for position in merged):
            return None
        return merged, end

    def walk(i: int, pos: int, spans: dict[str, _Span]) -> dict[str, _Span] | None:
        if not rendered.startswith(literals[i], pos):
            return None
        pos += len(literals[i])
        if i == len(pieces):
            return spans if pos == len(rendered) else None
        column, start, stop = pieces[i]
        if column in fixed:
            piece = fixed[column][start:stop]
            return walk(i + 1, pos + len(piece), spans) if rendered.startswith(piece, pos) else None
        # Shortest piece first; an empty one would be read back from CSV as null.
        ends = [pos + stop - start] if stop is not None else range(pos + 1, len(rendered) + 1)
        for end in ends:
            if end <= pos or end > len(rendered):
                continue
            span = place(spans.get(column), start, stop, rendered[pos:end])
            if span is not None and (found := walk(i + 1, end, {**spans, column: span})) is not None:
                return found
        return None

    spans = walk(0, 0, {})
    if spans is None:
        return None
    out = {}
    for column, (chars, end) in spans.items():
        length = end if end is not None else max(chars) + 1
        out[column] = "".join(chars.get(position, ASSEMBLY_PAD) for position in range(length))
    return out


def _generate_metadata_table(
    table: TablePlan,
    pools: dict[str, list],
    options: GenerationOptions,
    truncated: dict[str, int],
) -> pl.DataFrame:
    """Generate a ``_metadata`` dictionary file.

    Key columns take the Cartesian product of their pools so that every code the event tables can
    emit has a matching dictionary row. Anything else is descriptive filler.

    Args:
        table: The table plan.
        pools: Materialized pool values.
        options: The generation options.
        truncated: Accumulator recording how many key combinations were dropped, if any.

    Returns:
        The generated frame.
    """
    key_columns = [c for c in table.columns if c.covers_pool and c.pool_id]
    other_columns = [c for c in table.columns if c not in key_columns]

    if key_columns:
        # Every combination, not `table.n_rows` of them: the plan's estimate is based on requested
        # pool sizes, and truncating to it would leave part of the event vocabulary without a
        # dictionary entry — which shows up only as silently null descriptions in codes.parquet.
        combos = list(itertools.product(*[pools[c.pool_id] for c in key_columns]))
        if len(combos) > MAX_METADATA_ROWS:
            # Say so rather than truncating quietly: the dropped combinations become codes with no
            # description, which is indistinguishable from a broken metadata join unless flagged.
            truncated[table.prefix] = len(combos) - MAX_METADATA_ROWS
            logger.warning(
                f"{table.prefix}: {len(combos)} key combinations exceeds the {MAX_METADATA_ROWS}-row "
                f"cap; dropping {len(combos) - MAX_METADATA_ROWS}. Codes built from the dropped "
                f"combinations will have no description. Lower --vocab-size to fit."
            )
            combos = combos[:MAX_METADATA_ROWS]
    else:
        combos = [()] * table.n_rows

    data: dict[str, list] = {}
    for i, column in enumerate(key_columns):
        # Match Polars' event-code stringification (Python spells some floats differently).
        data[column.name] = pl.Series([combo[i] for combo in combos]).cast(pl.String).to_list()
    for column in other_columns:
        factory = ValueFactory(
            random.Random(f"{options.seed}:{table.prefix}:{column.name}"),
            null_fraction=0.0,
            numeric_ranges=options.numeric_ranges,
        )
        # Metadata files are read with `infer_schema=False`, so every column is text regardless of
        # what it looks like. Emitting strings here keeps generation honest about that.
        data[column.name] = [factory.text(column.name) for _ in combos]
    return pl.DataFrame(data, strict=False)


def _generate_column(
    column: ColumnPlan,
    table: TablePlan,
    plan: DatasetPlan,
    pools: dict[str, list],
    timelines: dict[int, SubjectTimeline],
    row_subjects: list[int],
    n_rows: int,
    options: GenerationOptions,
) -> list:
    """Generate the values for one column.

    Args:
        column: The column plan.
        table: The owning table plan.
        plan: The dataset plan.
        pools: Materialized pool values.
        timelines: Per-subject timelines.
        row_subjects: The subject id assigned to each row.
        n_rows: The number of rows.
        options: The generation options.

    Returns:
        The column's values.
    """
    rng = random.Random(f"{options.seed}:{table.prefix}:{column.name}")
    factory = ValueFactory(rng, null_fraction=options.null_fraction, numeric_ranges=options.numeric_ranges)
    constraint = column.constraint
    kind = constraint.kind

    strategy = options.column_generators.get((table.prefix, column.name))
    if strategy is not None:
        # Registered fragments of a timestamp share an instant per row, not independent samples.
        # Null injection uses the separate, per-column factory RNG and cannot shift this sequence.
        time_rng = random.Random(f"{options.seed}:{table.prefix}:registered-time")
        try:
            return _generate_temporal(
                column, timelines, row_subjects, n_rows, factory, time_rng, strategy=strategy
            )
        except Exception as e:
            raise ValueError(f"column generator for {(table.prefix, column.name)!r} failed: {e}") from e

    if column.pool_id == SUBJECT_POOL:
        return list(row_subjects)

    if column.pool_id == SUBJECT_KEY_POOL:
        # This column is hashed into the subject id, so it is a function *of the subject*, not a
        # free draw. Indexing by the row's subject keeps one key per subject (sampling would give
        # some subjects two keys and others none) and keeps the key consistent with the timeline
        # already chosen for that subject.
        pool = pools[column.pool_id]
        return [pool[(subject - 1) % len(pool)] for subject in row_subjects]

    if column.unique_in_table:
        # One distinct value per row: a repeated key on the target side of a plain left join fans
        # the join out, turning one referencing row into many.
        return [factory.token(column.name, i, constraint.min_chars) for i in range(n_rows)]

    if column.pool_id is not None:
        pool = pools[column.pool_id]
        if column.covers_pool:
            # Enumerate rather than sample: a key value missing from the target table would leave
            # every row referencing it unmatched.
            return [pool[i % len(pool)] for i in range(n_rows)]
        planned = plan.pools[column.pool_id]
        required = []
        if planned.kind in (ValueKind.NUMERIC, ValueKind.INTEGER):
            integral = planned.kind is ValueKind.INTEGER
            required = [
                n
                for n in (_as_number(v, integral) for v in planned.constraint.required_values)
                if n is not None
            ]
        # As for unshared numeric columns below: a uniform draw from a large pool almost never
        # lands on the literal a branch compares against, so required values get a fixed share.
        return [
            factory.maybe_null(
                required[i % len(required)]
                if required and rng.random() < REQUIRED_VALUE_SHARE
                else rng.choice(pool),
                constraint.nullable,
            )
            for i in range(n_rows)
        ]

    if kind in (ValueKind.DATETIME, ValueKind.DATE, ValueKind.DATETIME_STR):
        return _generate_temporal(column, timelines, row_subjects, n_rows, factory, rng)

    if kind in (ValueKind.NUMERIC, ValueKind.INTEGER, ValueKind.IDENTIFIER):
        integral = kind is not ValueKind.NUMERIC
        # A literal the config compares against (`$admissioncount == 1`) has to actually occur, or
        # the branch it gates never fires and the event yields nothing. About a third of rows take
        # a required value, so both sides of the comparison are represented.
        required = [n for n in (_as_number(v, integral) for v in constraint.required_values) if n is not None]
        return [
            factory.maybe_null(
                required[i % len(required)]
                if required and rng.random() < REQUIRED_VALUE_SHARE
                else factory.numeric(column.name, integral=integral, inferred_range=constraint.numeric_range),
                constraint.nullable,
            )
            for i in range(n_rows)
        ]
    if kind is ValueKind.BOOLEAN:
        return [factory.maybe_null(rng.choice([True, False]), constraint.nullable) for _ in range(n_rows)]
    if kind is ValueKind.STRING and not _has_shape_evidence(constraint):
        return [factory.maybe_null(factory.text(column.name), constraint.nullable) for _ in range(n_rows)]

    # Everything else becomes a small token vocabulary. UNKNOWN lands here because a column we
    # could not reason about is safest as an opaque string: it survives CSV inference unchanged and
    # can be interpolated into any code. A STRING column reaches here only when the config pins its
    # shape — a regex it must satisfy, or a literal it is compared against — because free-form
    # placeholder text would fail those.
    vocab = categorical_pool(column.name, constraint, options.vocab_size, rng)
    return [factory.maybe_null(rng.choice(vocab), constraint.nullable) for _ in range(n_rows)]


def _has_shape_evidence(constraint: ColumnConstraint) -> bool:
    """Return whether the config pins a column's textual shape.

    Args:
        constraint: The column's constraint.

    Returns:
        True if a regex or a compared-against literal constrains what the text must look like.

    Examples:
        >>> _has_shape_evidence(ColumnConstraint())
        False
        >>> _has_shape_evidence(ColumnConstraint(extract_patterns=("2003|2010",)))
        True
    """
    return bool(constraint.extract_patterns or constraint.match_patterns or constraint.required_values)


def _as_number(text: str, integral: bool) -> float | int | None:
    """Parse a recorded literal back into a number, or None if it is not numeric.

    Required values are recorded as text because most of them are categorical. When the column
    turns out to be numeric they have to be converted back.

    Args:
        text: The recorded literal.
        integral: Whether an integer is wanted.

    Returns:
        The number, or None if ``text`` is not numeric.

    Examples:
        >>> _as_number("1", True), _as_number("1.5", False), _as_number("alive", False)
        (1, 1.5, None)
    """
    try:
        return int(float(text)) if integral else float(text)
    except (TypeError, ValueError):
        return None


def _generate_temporal(
    column: ColumnPlan,
    timelines: dict[int, SubjectTimeline],
    row_subjects: list[int],
    n_rows: int,
    factory: ValueFactory,
    rng: random.Random,
    *,
    strategy: ColumnGenerator | None = None,
) -> list:
    """Generate a timestamp column, honoring birth/death roles and the recorded string formats.

    Args:
        column: The column plan.
        timelines: Per-subject timelines.
        row_subjects: The subject id assigned to each row.
        n_rows: The number of rows.
        factory: The value factory (for null injection).
        rng: The random source.
        strategy: Optional explicit renderer, applied instead of inferred formatting.

    Returns:
        The column's values — formatted strings when the config parses this column with
        ``strptime``, real datetimes otherwise.
    """
    constraint = column.constraint
    as_string = constraint.kind is ValueKind.DATETIME_STR
    out: list = []
    for i in range(n_rows):
        subject = row_subjects[i] if i < len(row_subjects) else None
        timeline = timelines.get(subject) if subject is not None else None
        if timeline is None:
            timeline = build_timeline(0, random.Random(f"anon:{i}"))

        role = constraint.effective_temporal_role
        if role == "birth":
            value = timeline.birth
        elif role == "death":
            value = timeline.death
        else:
            value = timeline.sample(rng)

        if value is None:
            out.append(None)
            continue
        if strategy is not None:
            rendered = strategy(value)
            if rendered is None and not constraint.effective_nullable:
                raise ValueError(f"returned null for non-nullable column at row {i}")
        else:
            rendered = format_datetime(value, constraint, i) if as_string else value
        out.append(factory.maybe_null(rendered, constraint.effective_nullable))
    return out

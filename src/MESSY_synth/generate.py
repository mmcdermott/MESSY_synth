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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from .constraints import ColumnConstraint, ValueKind
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
    """

    seed: int = 0
    n_subjects: int = 20
    rows_per_subject: int = DEFAULT_ROWS_PER_SUBJECT
    vocab_size: int = DEFAULT_VOCAB_SIZE
    null_fraction: float = 0.05
    death_probability: float = 0.3
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


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
    """

    plan: DatasetPlan
    frames: dict[str, pl.DataFrame]
    timelines: dict[int, SubjectTimeline]
    truncated_metadata: dict[str, int] = field(default_factory=dict)

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

    pools = _materialize_pools(plan, options)
    subject_ids: list[int] = pools[SUBJECT_POOL]
    timelines = {
        sid: build_timeline(sid, random.Random(f"{options.seed}:timeline:{sid}"), options.death_probability)
        for sid in subject_ids
    }

    joins = {t.input_prefix: t.join for t in cfg.event_tables if t.join is not None}
    frames: dict[str, pl.DataFrame] = {}
    truncated: dict[str, int] = {}
    for table in _generation_order(plan, joins):
        frame = _generate_table(table, plan, pools, timelines, subject_ids, options, truncated)
        join = joins.get(table.prefix)
        if join is not None and join.input_prefix in frames:
            frame = _adopt_join_keys(frame, table, join, frames[join.input_prefix], options)
        frames[table.prefix] = frame
    return GeneratedDataset(plan=plan, frames=frames, timelines=timelines, truncated_metadata=truncated)


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
        pools[pool_id] = categorical_pool(
            column, pool.constraint, pool.size, random.Random(f"{options.seed}:pool:{pool_id}")
        )
    return pools


def _generate_table(
    table: TablePlan,
    plan: DatasetPlan,
    pools: dict[str, list],
    timelines: dict[int, SubjectTimeline],
    subject_ids: list[int],
    options: GenerationOptions,
    truncated: dict[str, int],
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
    return pl.DataFrame(data, strict=False)


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
        data[column.name] = [combo[i] for combo in combos]
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
    del plan
    rng = random.Random(f"{options.seed}:{table.prefix}:{column.name}")
    factory = ValueFactory(rng, null_fraction=options.null_fraction, numeric_ranges=options.numeric_ranges)
    constraint = column.constraint
    kind = constraint.kind

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
        return [factory.maybe_null(rng.choice(pool), constraint.nullable) for _ in range(n_rows)]

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
) -> list:
    """Generate a timestamp column, honoring birth/death roles and the recorded string formats.

    Args:
        column: The column plan.
        timelines: Per-subject timelines.
        row_subjects: The subject id assigned to each row.
        n_rows: The number of rows.
        factory: The value factory (for null injection).
        rng: The random source.

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
        rendered = format_datetime(value, constraint, i) if as_string else value
        out.append(factory.maybe_null(rendered, constraint.effective_nullable))
    return out

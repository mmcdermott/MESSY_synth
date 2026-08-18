"""Turn inferred constraints into a concrete, self-consistent generation plan.

Inference (:mod:`MESSY_synth.inference`) answers "what type is each column?". This module answers
the harder question: "which columns must agree with each other?".

Structural validity is not per-column. A MEDS-Extract run only produces a non-trivial dataset if
values *line up* across files:

- every event table's subject column must draw from the **same** subject universe, or the extracted
  events never merge into shared subjects;
- a join key must hold values that actually exist on the far side, or the join drops every row;
- a ``_metadata`` dictionary's key column must hold the same text the event's ``code`` interpolates,
  or the metadata join silently matches nothing and the code descriptions come back null.

All three are the same constraint — *these columns share a value pool* — so they are handled by one
mechanism: a union-find over ``(table_prefix, column)`` pairs. Each resulting equivalence class
becomes a :class:`ValuePool` that generation materializes exactly once and every member column
draws from.

Row counts fall out of the same structure. A table that is the target of a join has to *cover* its
key pool (every value must appear at least once, or the referencing rows dangle), so its row count
is floored at the pool size and its key columns enumerate the pool rather than sampling it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dftly import Parser

from .constraints import ColumnConstraint, ConstraintSet, ValueKind
from .inference import infer_constraints

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MEDS_extract.config import MessyConfig

#: Pool identity for the one subject universe shared by every table in the dataset.
SUBJECT_POOL = "__subjects__"

#: Default number of rows generated per subject in an event table.
DEFAULT_ROWS_PER_SUBJECT = 4

#: Default number of distinct values in a categorical vocabulary.
DEFAULT_VOCAB_SIZE = 8

#: Upper bound on the Cartesian product used to populate a multi-key metadata dictionary.
MAX_METADATA_ROWS = 512


class UnionFind:
    """A minimal union-find over hashable keys, used to group columns that share a value pool.

    Examples:
        >>> uf = UnionFind()
        >>> uf.union(("labs", "stay_id"), ("stays", "stay_id"))
        >>> uf.find(("labs", "stay_id")) == uf.find(("stays", "stay_id"))
        True
        >>> uf.find(("other", "col")) == ("other", "col")
        True

        Grouping is transitive, so a chain of pairwise links collapses to one class:

        >>> uf.union(("stays", "stay_id"), ("icu", "stay_id"))
        >>> len({uf.find(k) for k in [("labs", "stay_id"), ("icu", "stay_id")]})
        1
    """

    def __init__(self) -> None:
        """Create an empty union-find with no elements."""
        self._parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(self, key: tuple[str, str]) -> tuple[str, str]:
        """Return the canonical representative of ``key``'s class.

        Args:
            key: The element to look up.

        Returns:
            The class representative.
        """
        parent = self._parent.setdefault(key, key)
        if parent != key:
            parent = self.find(parent)
            self._parent[key] = parent
        return parent

    def union(self, a: tuple[str, str], b: tuple[str, str]) -> None:
        """Merge the classes containing ``a`` and ``b``.

        Args:
            a: One element.
            b: The other element.
        """
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Order the representatives so the result is independent of call order, which keeps
            # generated pool ids (and therefore generated data) stable across runs.
            lo, hi = sorted([ra, rb])
            self._parent[hi] = lo


@dataclass(frozen=True)
class ValuePool:
    """A set of values shared by every column in one equivalence class.

    Attributes:
        pool_id: Stable identity, used as the random seed salt so a pool's values depend only on
            the config and the master seed, never on iteration order.
        kind: The merged :class:`~MESSY_synth.constraints.ValueKind` for the class.
        constraint: The merged constraint across every member column.
        size: How many distinct values the pool holds.
        members: The ``(prefix, column)`` pairs drawing from this pool, sorted.
    """

    pool_id: str
    kind: ValueKind
    constraint: ColumnConstraint
    size: int
    members: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ColumnPlan:
    """How one column of one file will be generated.

    Attributes:
        name: The column name as it must appear in the file.
        constraint: The inferred constraint for this column.
        pool_id: The :class:`ValuePool` this column draws from, if any.
        covers_pool: When True, this column enumerates its pool so that every pool value appears at
            least once. Set for the right-hand side of a join and for metadata key columns, where a
            missing value would leave referencing rows unmatched.
    """

    name: str
    constraint: ColumnConstraint
    pool_id: str | None = None
    covers_pool: bool = False


@dataclass(frozen=True)
class TablePlan:
    """How one source file will be generated.

    Attributes:
        prefix: The MESSY table prefix, e.g. ``nw_hosp/admissions``. Slashes become directories.
        columns: The columns to write, in a stable order.
        subject_column: The column holding subject ids, if this table has one.
        is_metadata: True for ``_metadata`` dictionary files. These are read by MEDS-Extract with
            ``infer_schema=False``, so every value must survive as literal text.
        is_subject_level: True when the table holds subject-level facts and should get exactly one
            row per subject.
        has_events: True when the config declares events on this table. False marks a pure join
            target — a dimension table that exists only to be looked up.
        n_rows: The number of rows to generate.
    """

    prefix: str
    columns: tuple[ColumnPlan, ...]
    subject_column: str | None
    is_metadata: bool
    is_subject_level: bool
    has_events: bool
    n_rows: int

    @property
    def kind(self) -> str:
        """Return a one-word description of this table's role.

        Returns:
            ``"metadata"``, ``"join-target"``, ``"subject-level"``, or ``"events"``.
        """
        if self.is_metadata:
            return "metadata"
        if not self.has_events:
            return "join-target"
        return "subject-level" if self.is_subject_level else "events"


@dataclass(frozen=True)
class DatasetPlan:
    """The complete generation plan for one MESSY config.

    Attributes:
        dataset_name: The dataset name declared in the config's ``etl:`` block.
        n_subjects: The size of the shared subject universe.
        tables: Every file to write.
        pools: Value pools keyed by pool id.
    """

    dataset_name: str
    n_subjects: int
    tables: tuple[TablePlan, ...]
    pools: dict[str, ValuePool] = field(default_factory=dict)

    def table(self, prefix: str) -> TablePlan:
        """Look up one table plan by prefix.

        Args:
            prefix: The table prefix.

        Returns:
            The matching plan.

        Raises:
            KeyError: If no table with that prefix is planned.
        """
        for t in self.tables:
            if t.prefix == prefix:
                return t
        raise KeyError(f"No planned table with prefix {prefix!r}. Planned: {[t.prefix for t in self.tables]}")


def build_plan(
    cfg: MessyConfig,
    *,
    n_subjects: int = 20,
    rows_per_subject: int = DEFAULT_ROWS_PER_SUBJECT,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> DatasetPlan:
    """Build the full generation plan for a MESSY config.

    Args:
        cfg: The parsed MESSY config.
        n_subjects: How many distinct subjects to invent.
        rows_per_subject: How many rows each non-subject-level event table gets per subject.
        vocab_size: How many distinct values each categorical vocabulary holds.

    Returns:
        The :class:`DatasetPlan`.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> cfg = MessyConfig.parse({
        ...     "_defaults": {"subject_id": "$patient_id"},
        ...     "patients": {"sex": {"code": 'f"SEX//{$sex}"', "time": None}},
        ...     "labs": {
        ...         "_table": {"join": {"stays": {"key": "stay_id", "cols": ["patient_id"]}}},
        ...         "lab": {"code": "$test", "time": '$t::"%Y-%m-%d"', "numeric_value": "$val"},
        ...     },
        ... })
        >>> plan = build_plan(cfg, n_subjects=10)
        >>> [t.prefix for t in plan.tables]
        ['labs', 'patients', 'stays']

        ``patients`` carries a static event, so it is subject-level: exactly one row per subject.

        >>> p = plan.table("patients")
        >>> p.is_subject_level, p.n_rows
        (True, 10)

        ``labs`` is a timed event table, so it gets several rows per subject:

        >>> plan.table("labs").n_rows
        40

        ``stays`` exists only because ``labs`` joins to it. It is sized to cover the key pool, and
        its key column enumerates that pool so no ``labs`` row dangles:

        >>> stays = plan.table("stays")
        >>> key = next(c for c in stays.columns if c.name == "stay_id")
        >>> key.covers_pool, stays.n_rows == plan.pools[key.pool_id].size
        (True, True)

        Both sides of the join share one pool, which is what makes the join land:

        >>> labs_key = next(c for c in plan.table("labs").columns if c.name == "stay_id")
        >>> labs_key.pool_id == key.pool_id
        True

        Every subject column across every table shares the single subject universe:

        >>> {c.pool_id for t in plan.tables for c in t.columns
        ...  if c.constraint.kind is ValueKind.SUBJECT_ID}
        {'__subjects__'}
    """
    constraints = infer_constraints(cfg)
    source_columns = cfg.needed_source_columns()
    metadata = _metadata_layout(cfg)

    uf = UnionFind()
    _link_subjects(constraints, source_columns, metadata, uf)
    _link_joins(cfg, uf)
    _link_metadata(metadata, uf)
    _link_identifiers_by_name(cfg, source_columns, uf)

    covering = _covering_columns(cfg, metadata)
    pools = _build_pools(constraints, source_columns, metadata, uf, n_subjects, vocab_size)

    tables: list[TablePlan] = []
    for prefix in sorted(source_columns):
        tables.append(
            _plan_table(
                prefix=prefix,
                column_names=source_columns[prefix],
                cfg=cfg,
                constraints=constraints,
                uf=uf,
                pools=pools,
                covering=covering,
                n_subjects=n_subjects,
                rows_per_subject=rows_per_subject,
                is_metadata=False,
            )
        )
    for prefix in sorted(metadata):
        tables.append(
            _plan_table(
                prefix=prefix,
                column_names=sorted(metadata[prefix].file_columns),
                cfg=cfg,
                constraints=constraints,
                uf=uf,
                pools=pools,
                covering=covering,
                n_subjects=n_subjects,
                rows_per_subject=rows_per_subject,
                is_metadata=True,
                metadata_keys=metadata[prefix].key_source_columns,
            )
        )

    return DatasetPlan(
        dataset_name=_dataset_name(cfg),
        n_subjects=n_subjects,
        tables=tuple(sorted(tables, key=lambda t: t.prefix)),
        pools=pools,
    )


def _dataset_name(cfg: MessyConfig) -> str:
    """Return the config's dataset name, tolerating configs that omit one.

    ``MessyConfig.dataset_name`` raises when a path-resolved spec has no ``etl.dataset_name``. That
    is the right rule for *running* an ETL, but synthesis only needs a label, so a config written
    purely as an event-conversion spec should still be usable here.

    Args:
        cfg: The parsed MESSY config.

    Returns:
        The declared dataset name, or ``"synthetic"`` if the config declares none.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> _dataset_name(MessyConfig.parse({"t": {"e": {"code": "X", "time": None}}}))
        'synthetic'
        >>> _dataset_name(MessyConfig.parse({
        ...     "etl": {"dataset_name": "Demo"}, "t": {"e": {"code": "X", "time": None}},
        ... }))
        'Demo'
    """
    try:
        return cfg.dataset_name
    except ValueError:
        return "synthetic"


@dataclass
class _MetadataLayout:
    """What one ``_metadata`` dictionary file must contain.

    Attributes:
        file_columns: Every column the block's expressions read from the dictionary file.
        key_source_columns: Dictionary-file columns that act as join keys, mapped to the
            ``(event_prefix, code_component_column)`` pairs they must agree with.
    """

    file_columns: set[str] = field(default_factory=set)
    key_source_columns: dict[str, set[tuple[str, str]]] = field(default_factory=dict)


def _metadata_layout(cfg: MessyConfig) -> dict[str, _MetadataLayout]:
    """Work out the column set and join keys of every ``_metadata`` dictionary file.

    A metadata block maps output names to dftly expressions over the *dictionary file's* columns.
    An output name that coincides with one of the declaring event's code components is a join key;
    the expression under it names the dictionary column that must carry the matching text.

    Args:
        cfg: The parsed MESSY config.

    Returns:
        A mapping of metadata prefix to its layout.
    """
    parser = Parser()
    code_components: dict[str, tuple[str, set[str]]] = {}
    for table in cfg.event_tables:
        for event in table.events:
            code_components[f"{table.input_prefix}/{event.name}"] = (
                table.input_prefix,
                set(event.code_source_columns),
            )

    out: dict[str, _MetadataLayout] = {}
    for prefix, entries in cfg.events_by_metadata_prefix().items():
        layout = out.setdefault(prefix, _MetadataLayout())
        for entry in entries:
            event_prefix, components = code_components.get(entry["source_block"], ("", set()))
            for out_name, expr in entry["_metadata"].items():
                try:
                    node = parser(expr)
                except Exception:  # pragma: no cover - malformed configs fail earlier
                    continue
                referenced = set(node.referenced_columns)
                layout.file_columns |= referenced
                if out_name in components:
                    # A key output is canonically `key: $key`; use the column it actually reads.
                    for col in referenced or {out_name}:
                        layout.key_source_columns.setdefault(col, set()).add((event_prefix, out_name))
    return out


def _link_subjects(
    constraints: ConstraintSet,
    source_columns: dict[str, list[str]],
    metadata: dict[str, _MetadataLayout],
    uf: UnionFind,
) -> None:
    """Union every subject-id column in the dataset into the one shared subject universe.

    Args:
        constraints: The inferred constraints.
        source_columns: MEDS-Extract's per-prefix column plan.
        metadata: Metadata layouts (never carry subject ids, but kept for symmetry).
        uf: The union-find to write into.
    """
    del metadata
    for prefix, columns in source_columns.items():
        for column in columns:
            if constraints.get(prefix, column).kind is ValueKind.SUBJECT_ID:
                uf.union((SUBJECT_POOL, SUBJECT_POOL), (prefix, column))


def _link_joins(cfg: MessyConfig, uf: UnionFind) -> None:
    """Union each join's left key with its right key.

    Args:
        cfg: The parsed MESSY config.
        uf: The union-find to write into.
    """
    for table in cfg.event_tables:
        if table.join is None:
            continue
        for left, right in zip(table.join.left_on, table.join.right_on, strict=True):
            uf.union((table.input_prefix, left), (table.join.input_prefix, right))


def _link_metadata(metadata: dict[str, _MetadataLayout], uf: UnionFind) -> None:
    """Union each metadata key column with the event code component it must match.

    Args:
        metadata: The metadata layouts.
        uf: The union-find to write into.
    """
    for prefix, layout in metadata.items():
        for meta_col, targets in layout.key_source_columns.items():
            for event_prefix, event_col in targets:
                if event_prefix:
                    uf.union((prefix, meta_col), (event_prefix, event_col))


def _link_identifiers_by_name(
    cfg: MessyConfig,
    source_columns: dict[str, list[str]],
    uf: UnionFind,
) -> None:
    """Union same-named identifier columns across tables.

    A column that acts as a join key somewhere (``hadm_id``, ``stay_id``) usually appears on other
    tables too, as a passthrough MEDS output column. The config never states that those are the
    same identifier — the join only declares the two tables it links — but generating them from
    unrelated pools would produce data where an event's ``hadm_id`` names an admission that does
    not exist. Linking by name keeps the synthetic dataset internally coherent, which matters for
    the "demonstrate an ETL" use case even though the ETL would run either way.

    Args:
        cfg: The parsed MESSY config.
        source_columns: MEDS-Extract's per-prefix column plan.
        uf: The union-find to write into.
    """
    key_names = {
        key
        for table in cfg.event_tables
        if table.join is not None
        for key in (*table.join.left_on, *table.join.right_on)
    }
    for prefix, columns in source_columns.items():
        for column in columns:
            if column in key_names:
                uf.union((f"__name__{column}", column), (prefix, column))


def _covering_columns(cfg: MessyConfig, metadata: dict[str, _MetadataLayout]) -> set[tuple[str, str]]:
    """Return the columns that must enumerate their whole pool.

    Args:
        cfg: The parsed MESSY config.
        metadata: The metadata layouts.

    Returns:
        ``(prefix, column)`` pairs on the receiving end of a join or metadata lookup.
    """
    covering: set[tuple[str, str]] = set()
    for table in cfg.event_tables:
        if table.join is not None:
            for right in table.join.right_on:
                covering.add((table.join.input_prefix, right))
    for prefix, layout in metadata.items():
        for meta_col in layout.key_source_columns:
            covering.add((prefix, meta_col))
    return covering


def _build_pools(
    constraints: ConstraintSet,
    source_columns: dict[str, list[str]],
    metadata: dict[str, _MetadataLayout],
    uf: UnionFind,
    n_subjects: int,
    vocab_size: int,
) -> dict[str, ValuePool]:
    """Materialize one :class:`ValuePool` per equivalence class that needs shared values.

    Args:
        constraints: The inferred constraints.
        source_columns: MEDS-Extract's per-prefix column plan.
        metadata: The metadata layouts.
        uf: The populated union-find.
        n_subjects: Size of the subject universe.
        vocab_size: Size of each categorical vocabulary.

    Returns:
        A mapping of pool id to pool.
    """
    members: dict[tuple[str, str], list[tuple[str, str]]] = {}
    all_columns = [(p, c) for p, cols in source_columns.items() for c in cols]
    all_columns += [(p, c) for p, layout in metadata.items() for c in sorted(layout.file_columns)]

    for key in all_columns:
        members.setdefault(uf.find(key), []).append(key)

    pools: dict[str, ValuePool] = {}
    for root, group in members.items():
        merged = ColumnConstraint()
        for prefix, column in group:
            merged = merged.merge(constraints.get(prefix, column))
        is_subject = root == (SUBJECT_POOL, SUBJECT_POOL) or merged.kind is ValueKind.SUBJECT_ID
        shared = len(group) > 1 or merged.kind in (ValueKind.CATEGORICAL, ValueKind.IDENTIFIER)
        if not (is_subject or shared):
            continue
        if is_subject:
            pool_id, size, kind = SUBJECT_POOL, n_subjects, ValueKind.SUBJECT_ID
        else:
            pool_id = "|".join(f"{p}.{c}" for p, c in sorted(group))
            kind = merged.kind if merged.kind is not ValueKind.UNKNOWN else ValueKind.CATEGORICAL
            size = vocab_size
        pools[pool_id] = ValuePool(
            pool_id=pool_id,
            kind=kind,
            constraint=merged,
            size=size,
            members=tuple(sorted(group)),
        )
    return pools


def _pool_for(key: tuple[str, str], uf: UnionFind, pools: dict[str, ValuePool]) -> str | None:
    """Find the pool id a column belongs to, if any.

    Args:
        key: The ``(prefix, column)`` pair.
        uf: The populated union-find.
        pools: The materialized pools.

    Returns:
        The pool id, or None if this column is generated independently per row.
    """
    root = uf.find(key)
    if root == (SUBJECT_POOL, SUBJECT_POOL):
        return SUBJECT_POOL
    for pool_id, pool in pools.items():
        if key in pool.members:
            return pool_id
    return None


def _is_subject_level(cfg: MessyConfig, prefix: str) -> bool:
    """Decide whether a table holds subject-level facts (one row per subject).

    The signal is a *static* event — one declared with ``time: null``. A static event is a fact
    about the subject rather than about a moment, so a table that declares one is subject-scoped;
    generating several rows for the same subject would emit that fact several times over. Birth and
    death events count for the same reason: they are once-per-subject by construction.

    Args:
        cfg: The parsed MESSY config.
        prefix: The table prefix to test.

    Returns:
        True if the table should get exactly one row per subject.
    """
    for table in cfg.event_tables:
        if table.input_prefix != prefix:
            continue
        for event in table.events:
            if event.columns.get("time") is None:
                return True
            if (event.raw_code or "") in ("MEDS_BIRTH", "MEDS_DEATH"):
                return True
    return False


def _plan_table(
    *,
    prefix: str,
    column_names: list[str],
    cfg: MessyConfig,
    constraints: ConstraintSet,
    uf: UnionFind,
    pools: dict[str, ValuePool],
    covering: set[tuple[str, str]],
    n_subjects: int,
    rows_per_subject: int,
    is_metadata: bool,
    metadata_keys: dict[str, set[tuple[str, str]]] | None = None,
) -> TablePlan:
    """Assemble the plan for one file.

    Args:
        prefix: The table prefix.
        column_names: The columns this file must carry.
        cfg: The parsed MESSY config.
        constraints: The inferred constraints.
        uf: The populated union-find.
        pools: The materialized pools.
        covering: Columns that must enumerate their pool.
        n_subjects: The subject-universe size.
        rows_per_subject: Rows per subject for timed event tables.
        is_metadata: Whether this is a ``_metadata`` dictionary file.
        metadata_keys: For metadata files, the key columns and what they must match.

    Returns:
        The assembled :class:`TablePlan`.
    """
    columns: list[ColumnPlan] = []
    subject_column: str | None = None
    for name in column_names:
        constraint = constraints.get(prefix, name)
        pool_id = _pool_for((prefix, name), uf, pools)
        if constraint.kind is ValueKind.SUBJECT_ID:
            subject_column = name
        columns.append(
            ColumnPlan(
                name=name,
                constraint=constraint,
                pool_id=pool_id,
                covers_pool=(prefix, name) in covering,
            )
        )

    has_events = any(t.input_prefix == prefix for t in cfg.event_tables)

    if is_metadata:
        keys = sorted(metadata_keys or {})
        n_rows = 1
        for key in keys:
            pool_id = _pool_for((prefix, key), uf, pools)
            n_rows *= pools[pool_id].size if pool_id else 1
        n_rows = max(1, min(n_rows, MAX_METADATA_ROWS))
        subject_level = False
    else:
        subject_level = _is_subject_level(cfg, prefix)
        if not has_events:
            # A pure join target — a dimension table that exists only to be looked up. It needs one
            # row per key value and nothing more; extra rows would just be unreachable.
            n_rows = max(
                (pools[c.pool_id].size for c in columns if c.covers_pool and c.pool_id),
                default=n_subjects,
            )
        else:
            n_rows = n_subjects if subject_level else n_subjects * rows_per_subject
        # Never emit fewer rows than a pool this table must cover.
        for column in columns:
            if column.covers_pool and column.pool_id:
                n_rows = max(n_rows, pools[column.pool_id].size)

    # Lead with the subject id: it is the column a reader looks for first, and it makes the
    # shared-universe relationship between files visible at a glance.
    columns.sort(key=lambda c: (c.name != subject_column, c.name))

    return TablePlan(
        prefix=prefix,
        columns=tuple(columns),
        subject_column=subject_column,
        is_metadata=is_metadata,
        is_subject_level=subject_level,
        has_events=has_events,
        n_rows=n_rows,
    )

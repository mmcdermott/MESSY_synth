"""Infer a synthetic source schema by walking the dftly expressions in a MESSY config.

A MEDS-Extract v0.7 MESSY config describes an ETL as expressions *over* raw source columns. It
never declares those columns' types — the raw files are supposed to already exist. To synthesize
inputs we have to run that reasoning backwards: read the expressions, and deduce what the files
must have contained for the expressions to make sense.

The algorithm is bidirectional type propagation over the dftly AST. Every MEDS output slot carries
a known expectation (``time`` must yield a timestamp, ``numeric_value`` a number, ``code`` a
string), and each node type says how its own expectation constrains its children. Walking down to
the leaves turns "``time: coalesce($dod::?"%Y-%m-%d %H:%M:%S", $dod::?"%Y-%m-%d")``" into "column
``dod`` is a string written in one of two date formats".

Column references are not always leaves. Three cases are resolved during the walk:

- a name declared in ``_table.cols`` is *derived*, so the walk recurses into its defining
  expression, carrying the expectation through to the raw columns underneath;
- a name pulled in by ``_table.join`` belongs to the **join target's** file, so the constraint is
  recorded against that other table;
- anything else is a real column of this table's own file.

What this module does *not* decide is which columns exist. That comes from MEDS-Extract itself, via
``MessyConfig.needed_source_columns()`` — the same call the ``convert_to_parquet`` stage uses to
decide what to read. Inference only supplies types. Keeping those two concerns apart means a column
we fail to reason about still gets generated (as a plain string) rather than silently going
missing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dftly.nodes import (
    Cast,
    Coalesce,
    Conditional,
    Hash,
    LenChars,
    RegexExtract,
    RegexMatch,
    SetTime,
    SignedHash,
    Split,
    StringInterpolate,
    Strptime,
    Substring,
)
from dftly.nodes.base import Column, Literal, NodeBase
from dftly.nodes.datetime import _DtAccessor
from dftly.nodes.types import TYPES

from .constraints import ColumnConstraint, ConstraintSet, ValueKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from MEDS_extract.config import MessyConfig, TableConfig

logger = logging.getLogger(__name__)

#: Expectation imposed by each MEDS output slot. Anything not listed (a passthrough column such as
#: ``hadm_id``) carries no expectation and falls through to the default string treatment.
OUTPUT_SLOT_KINDS: dict[str, ValueKind] = {
    "code": ValueKind.CATEGORICAL,
    "time": ValueKind.DATETIME,
    "numeric_value": ValueKind.NUMERIC,
    "text_value": ValueKind.STRING,
}

#: MEDS reserves two codes for the bounds of a subject's timeline. A column feeding either one's
#: ``time`` slot is placed deliberately during generation instead of being sampled like any other
#: timestamp, so births land before the events they precede and deaths land after them.
MEDS_TEMPORAL_ROLES: dict[str, str] = {"MEDS_BIRTH": "birth", "MEDS_DEATH": "death"}

#: Plausible calendar-year bounds for the left term of date-producing integer arithmetic.
YEAR_RANGE = (1950.0, 2015.0)

#: Plausible bounds for the right term — an age or year offset.
YEAR_OFFSET_RANGE = (0.0, 90.0)

#: dftly cast targets whose *source* must be a number. Duration units are the common case: the 0.7
#: idiom for offset-encoded datasets is ``$origin + $offset::minutes``, where ``offset`` is numeric.
_NUMERIC_SOURCE_CASTS = frozenset(
    {
        "days",
        "hours",
        "microseconds",
        "milliseconds",
        "minutes",
        "months",
        "nanoseconds",
        "seconds",
        "weeks",
        "years",
        "duration",
        "year",
    }
)

#: dftly cast targets that produce a number, so their source is number-like too.
_NUMERIC_TARGET_CASTS = frozenset(
    {
        "double",
        "float",
        "float32",
        "float64",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "int128",
        "integer",
        "long",
        "uint",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)


def infer_constraints(cfg: MessyConfig) -> ConstraintSet:
    """Walk every expression in a MESSY config and collect per-column constraints.

    Args:
        cfg: A parsed MESSY config.

    Returns:
        The accumulated :class:`~MESSY_synth.constraints.ConstraintSet`, keyed by
        ``(table_prefix, column_name)``.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> cfg = MessyConfig.parse({
        ...     "_defaults": {"subject_id": "$patient_id"},
        ...     "labs": {
        ...         "lab": {
        ...             "code": 'f"LAB//{$test_name}"',
        ...             "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
        ...             "numeric_value": "$result",
        ...         },
        ...     },
        ... })
        >>> cs = infer_constraints(cfg)

        The subject-id column is pinned to the strict integer cast MEDS-Extract applies:

        >>> cs.get("labs", "patient_id").kind.name
        'SUBJECT_ID'

        A ``strptime`` source becomes a string with a recorded format:

        >>> c = cs.get("labs", "charttime")
        >>> c.kind.name, c.datetime_formats, c.strict_parse
        ('DATETIME_STR', ('%Y-%m-%d %H:%M:%S',), True)

        An f-string component becomes a categorical, and ``numeric_value`` a number:

        >>> cs.get("labs", "test_name").kind.name, cs.get("labs", "result").kind.name
        ('CATEGORICAL', 'NUMERIC')

        Columns pulled through a join are attributed to the table they actually live in:

        >>> cfg = MessyConfig.parse({
        ...     "labs": {
        ...         "_defaults": {"subject_id": "$patient_id"},
        ...         "_table": {"join": {"stays": {"key": "stay_id", "cols": ["dischtime"]}}},
        ...         "lab": {"code": "$test", "time": '$dischtime::"%Y-%m-%d"'},
        ...     },
        ... })
        >>> cs = infer_constraints(cfg)
        >>> cs.get("stays", "dischtime").datetime_formats
        ('%Y-%m-%d',)

        Join keys are non-nullable identifiers on *both* sides, since a null key joins to nothing:

        >>> cs.get("labs", "stay_id").kind.name, cs.get("labs", "stay_id").nullable
        ('IDENTIFIER', False)
        >>> cs.get("stays", "stay_id").nullable
        False

        A ``_table.cols`` entry is derived rather than read, so its expectation is carried through
        to the raw columns it is computed from:

        >>> cfg = MessyConfig.parse({
        ...     "vitals": {
        ...         "_defaults": {"subject_id": "$pid"},
        ...         "_table": {"cols": {
        ...             "_origin": '("2016-01-01"::?"%Y-%m-%d")::datetime',
        ...             "chart_t": "$_origin + $offset::minutes",
        ...         }},
        ...         "v": {"code": "HR", "time": "$chart_t", "numeric_value": "$val"},
        ...     },
        ... })
        >>> cs = infer_constraints(cfg)
        >>> cs.get("vitals", "offset").kind.name
        'NUMERIC'

        The derived name itself is never treated as a source column:

        >>> sorted(cs.columns_for("vitals"))
        ['offset', 'pid', 'val']
    """
    cs = ConstraintSet()
    for table in cfg.event_tables:
        _infer_table(table, cs)
    _infer_metadata(cfg, cs)
    return cs


def _infer_table(table: TableConfig, cs: ConstraintSet) -> None:
    """Collect constraints for one event table, its join target, and its events.

    Args:
        table: The table config to walk.
        cs: The accumulator to write into.
    """
    ctx = _Context(prefix=table.input_prefix, derived=dict(table.cols), join=table.join)

    if table.subject_id_node is not None:
        _visit(table.subject_id_node, ValueKind.SUBJECT_ID, ctx, cs, "subject_id")
    else:
        cs.observe(
            table.input_prefix,
            "subject_id",
            ColumnConstraint(kind=ValueKind.SUBJECT_ID, nullable=False).with_note("implicit subject_id"),
        )

    if table.join is not None:
        key_constraint = ColumnConstraint(kind=ValueKind.IDENTIFIER, nullable=False)
        for left, right in zip(table.join.left_on, table.join.right_on, strict=True):
            # A left key may itself be a `_table.cols` literal used to filter the right side (the
            # canonical `drug_type: "'MAIN'"` idiom); those are derived, not read, so route the
            # observation through the normal resolver rather than asserting a source column.
            if left in ctx.derived:
                _visit(
                    ctx.derived[left], ValueKind.IDENTIFIER, ctx, cs, f"join key -> {table.join.input_prefix}"
                )
            else:
                cs.observe(
                    table.input_prefix,
                    left,
                    key_constraint.with_note(f"join key -> {table.join.input_prefix}"),
                )
            cs.observe(
                table.join.input_prefix,
                right,
                key_constraint.with_note(f"join key <- {table.input_prefix}"),
            )
        # Aggregated joins (`cols: {deathtime: min}`) reduce the named column, so it must be
        # orderable; min/max over strings works in polars, so no numeric constraint is forced.
        for col, agg in table.join.aggregations:
            cs.observe(
                table.join.input_prefix,
                col,
                ColumnConstraint().with_note(f"aggregated with {agg}() for {table.input_prefix}"),
            )

    for event in table.events:
        role = MEDS_TEMPORAL_ROLES.get((event.raw_code or "").strip())
        for slot, node in event.columns.items():
            if node is None:
                continue
            expect = OUTPUT_SLOT_KINDS.get(slot, ValueKind.UNKNOWN)
            _visit(node, expect, ctx, cs, f"{event.name}.{slot}")
            if role is not None and slot == "time":
                _decorate(
                    node,
                    ctx,
                    cs,
                    ColumnConstraint(temporal_role=role).with_note(f"{event.name}: {role} timestamp"),
                    frozenset(),
                )


def _infer_metadata(cfg: MessyConfig, cs: ConstraintSet) -> None:
    """Collect constraints for every ``_metadata`` dictionary file.

    Metadata files are the one input MEDS-Extract reads with ``infer_schema=False``, so every
    column arrives as a string and the join keys must match the event side's rendered text
    exactly. Both sides are therefore recorded as categorical.

    Args:
        cfg: The parsed MESSY config.
        cs: The accumulator to write into.
    """
    for prefix, entries in cfg.events_by_metadata_prefix().items():
        for entry in entries:
            block = entry["_metadata"]
            for out_name, expr in block.items():
                node = _parse(expr)
                if node is None:
                    continue
                ctx = _Context(prefix=prefix, derived={}, join=None)
                _visit(node, ValueKind.CATEGORICAL, ctx, cs, f"_metadata[{prefix}].{out_name}")


def _parse(expr: object) -> NodeBase | None:
    """Parse a raw ``_metadata`` expression into a dftly node, tolerating already-parsed input.

    Args:
        expr: A dftly expression string, or an already-parsed node.

    Returns:
        The parsed node, or None if it cannot be parsed.

    Examples:
        >>> _parse("$foo")
        Column('foo')
        >>> _parse(None) is None
        True
    """
    if expr is None:
        return None
    if isinstance(expr, NodeBase):
        return expr
    from dftly import Parser

    try:
        return Parser()(expr)
    except Exception as e:  # pragma: no cover - defensive; malformed configs fail earlier
        logger.debug(f"Could not parse metadata expression {expr!r}: {e}")
        return None


class _Context:
    """Where a walk currently is: which table, and how to resolve a bare column name.

    Attributes:
        prefix: The source prefix that unqualified column references belong to.
        derived: ``_table.cols`` entries, which are computed rather than read.
        join: This table's join, if any — its ``cols`` name the columns that live on the far side.
    """

    __slots__ = ("derived", "join", "prefix")

    def __init__(self, prefix: str, derived: dict[str, NodeBase], join: object) -> None:
        self.prefix = prefix
        self.derived = derived
        self.join = join

    def owner_of(self, column: str) -> str:
        """Return the source prefix that ``column`` is read from.

        Args:
            column: The referenced column name.

        Returns:
            The join target's prefix if the column is pulled through the join, else this table's.
        """
        if self.join is not None and column in self.join.cols:
            return self.join.input_prefix
        return self.prefix


def _visit(
    node: NodeBase,
    expect: ValueKind,
    ctx: _Context,
    cs: ConstraintSet,
    note: str,
    seen: frozenset[str] = frozenset(),
    range_hint: tuple[float, float] | None = None,
) -> None:
    """Propagate an expectation down one dftly expression, recording what the leaves must hold.

    Args:
        node: The node to visit.
        expect: What this node's value is required to be.
        ctx: The table context used to resolve column references.
        cs: The accumulator to write into.
        note: Provenance text attached to every constraint recorded beneath this node.
        seen: Derived-column names already being expanded on this path, guarding against a
            ``_table.cols`` entry that (directly or transitively) references itself.
        range_hint: Numeric bounds the surrounding expression implies. Set when a cast or a parse
            reveals a magnitude the value kind cannot express — ``::year`` and ``%Y`` both mean
            "this number is a calendar year", which the default 1-100 range would violate.
    """
    match node:
        case Literal():
            return

        case Column():
            _visit_column(node, expect, ctx, cs, note, seen, range_hint)

        case Strptime():
            fmt = _literal_value(node.kwargs.get("format"))
            strict = _literal_value(node.kwargs.get("strict"))
            source = node.kwargs.get("source")
            if source is not None:
                _visit(
                    source,
                    ValueKind.DATETIME_STR,
                    ctx,
                    cs,
                    note,
                    seen,
                )
                _decorate(
                    source,
                    ctx,
                    cs,
                    ColumnConstraint(
                        kind=ValueKind.DATETIME_STR,
                        datetime_formats=(fmt,) if isinstance(fmt, str) else (),
                        # `::` is strict, `::?` is lenient. dftly records the lenient form by
                        # setting strict=False; an absent kwarg means the strict default.
                        strict_parse=strict is not False,
                        nullable=strict is not None and strict is not True,
                    ).with_note(f"{note}: strptime({fmt!r})"),
                    seen,
                )

        case Cast():
            _visit_cast(node, expect, ctx, cs, note, seen, range_hint)

        case StringInterpolate():
            # args[0] is the literal template; the rest are the interpolated pieces.
            for part in node.args[1:]:
                _visit(part, ValueKind.CATEGORICAL, ctx, cs, f"{note}: f-string component", seen)

        case Coalesce():
            for arg in node.args:
                _visit(arg, expect, ctx, cs, note, seen, range_hint)
            # A `?? 'UNK'` fallback is a value the slot can genuinely take, so it belongs in the
            # column's pool. That matters for a code component: without it the dictionary the code
            # joins against never contains 'UNK', and every row that fell back gets a null
            # description — a silent gap in the extracted metadata.
            fallbacks = tuple(
                str(v) for a in node.args if (v := _literal_value(a)) is not None and not isinstance(v, bool)
            )
            if fallbacks:
                for arg in node.args:
                    if isinstance(arg, Literal):
                        continue
                    _decorate(
                        arg,
                        ctx,
                        cs,
                        ColumnConstraint(required_values=fallbacks).with_note(
                            f"{note}: coalesce fallback {fallbacks}"
                        ),
                        seen,
                    )

        case Conditional():
            if (when := node.kwargs.get("when")) is not None:
                _visit(when, ValueKind.BOOLEAN, ctx, cs, f"{note}: condition", seen)
            for branch in ("then", "otherwise"):
                if (b := node.kwargs.get(branch)) is not None:
                    _visit(b, expect, ctx, cs, note, seen)

        case RegexMatch() | RegexExtract():
            pattern = _literal_value(node.kwargs.get("pattern"))
            source = node.kwargs.get("source")
            if source is not None:
                _visit(source, ValueKind.STRING, ctx, cs, note, seen)
                # A *test* (`/re/ in $col`) gates a branch, so a mix of matching and non-matching
                # values is what exercises the config. An *extract* produces the value itself: a
                # miss yields null and usually poisons every column derived from it, so those
                # patterns must always match.
                patterns = (pattern,) if isinstance(pattern, str) else ()
                is_extract = isinstance(node, RegexExtract)
                _decorate(
                    source,
                    ctx,
                    cs,
                    ColumnConstraint(
                        kind=ValueKind.STRING,
                        match_patterns=() if is_extract else patterns,
                        extract_patterns=patterns if is_extract else (),
                    ).with_note(f"{note}: {'extract' if is_extract else 'match'} {pattern!r}"),
                    seen,
                )

        case Substring():
            source = node.kwargs.get("source")
            stop = _literal_value(node.kwargs.get("stop"))
            start = _literal_value(node.kwargs.get("start"))
            need = max(v for v in (stop, start, 0) if isinstance(v, int))
            if source is not None:
                _visit(source, ValueKind.STRING, ctx, cs, note, seen)
                _decorate(
                    source,
                    ctx,
                    cs,
                    ColumnConstraint(kind=ValueKind.STRING, min_chars=need).with_note(
                        f"{note}: sliced [{start}:{stop}]"
                    ),
                    seen,
                )

        case LenChars() | Split():
            for child in _children(node):
                _visit(child, ValueKind.STRING, ctx, cs, note, seen)

        case Hash() | SignedHash():
            # A hash accepts anything and yields the subject id itself, so the source column is a
            # free-form identifier rather than something that must cast to Int64.
            for child in _children(node):
                _visit(child, ValueKind.IDENTIFIER, ctx, cs, f"{note}: hashed", seen)

        case SetTime():
            if node.args:
                _visit(node.args[0], ValueKind.DATETIME, ctx, cs, note, seen)

        case _DtAccessor():
            for child in _children(node):
                _visit(child, ValueKind.DATETIME, ctx, cs, note, seen)

        case _:
            _visit_generic(node, expect, ctx, cs, note, seen, range_hint)


def _visit_column(
    node: Column,
    expect: ValueKind,
    ctx: _Context,
    cs: ConstraintSet,
    note: str,
    seen: frozenset[str],
    range_hint: tuple[float, float] | None = None,
) -> None:
    """Record (or forward) the expectation on a column reference.

    Args:
        node: The column node.
        expect: The expectation to record.
        ctx: The resolving context.
        cs: The accumulator.
        note: Provenance text.
        seen: Derived names already being expanded, for cycle safety.
        range_hint: Numeric bounds implied by the surrounding expression.
    """
    name = node.args[0]
    if name in ctx.derived and name not in seen:
        _visit(ctx.derived[name], expect, ctx, cs, f"{note} via {name}", seen | {name}, range_hint)
        return
    if name in ctx.derived:
        return  # cyclic `_table.cols` reference; MEDS-Extract rejects these, so just stop.
    cs.observe(
        ctx.owner_of(name),
        name,
        ColumnConstraint(
            kind=expect,
            numeric_range=range_hint,
            nullable=expect not in (ValueKind.SUBJECT_ID, ValueKind.IDENTIFIER),
        ).with_note(note),
    )


def _visit_cast(
    node: Cast,
    expect: ValueKind,
    ctx: _Context,
    cs: ConstraintSet,
    note: str,
    seen: frozenset[str],
    range_hint: tuple[float, float] | None = None,
) -> None:
    """Map a ``::`` cast target onto the expectation its source column must satisfy.

    Args:
        node: The cast node.
        expect: The expectation on the cast's own result (mostly unused; the target decides).
        ctx: The resolving context.
        cs: The accumulator.
        note: Provenance text.
        seen: Derived names already being expanded.
        range_hint: Numeric bounds implied by the surrounding expression.
    """
    target = _literal_value(node.kwargs.get("type"))
    source = node.kwargs.get("source")
    if source is None:
        return
    if target == "year":
        # `$n::year` reads an integer as a calendar year, so the number underneath has to look like
        # one. MIMIC-IV's `($anchor_year - $anchor_age)::str` then `::year` is the canonical case:
        # with the default numeric range the difference is a two-digit nonsense year and every
        # birth event is dropped.
        _visit(source, ValueKind.NUMERIC, ctx, cs, f"{note}: ::year", seen, YEAR_RANGE)
        return
    if isinstance(target, str) and target in _NUMERIC_SOURCE_CASTS | _NUMERIC_TARGET_CASTS:
        child_expect = ValueKind.NUMERIC
    elif target in ("date", "datetime", "time"):
        child_expect = ValueKind.DATETIME
    elif target in ("str", "string", "utf8"):
        # `::str` is a rendering step. It normally says nothing about the input, but under a
        # DATETIME_STR expectation it is the `($anchor_year - $anchor_age)::str` idiom, whose
        # rendered result gets parsed as a date — so the expectation has to survive the cast for
        # the arithmetic underneath to be typed correctly.
        if expect is ValueKind.DATETIME_STR:
            child_expect = expect
        else:
            child_expect = ValueKind.UNKNOWN if expect >= ValueKind.NUMERIC else expect
    elif target in ("bool", "boolean"):
        child_expect = ValueKind.BOOLEAN
    elif isinstance(target, str) and target in TYPES:
        child_expect = ValueKind.UNKNOWN
    else:
        child_expect = expect
    _visit(source, child_expect, ctx, cs, f"{note}: ::{target}", seen, range_hint)


def _visit_generic(
    node: NodeBase,
    expect: ValueKind,
    ctx: _Context,
    cs: ConstraintSet,
    note: str,
    seen: frozenset[str],
    range_hint: tuple[float, float] | None = None,
) -> None:
    """Handle arithmetic, comparison, and boolean nodes uniformly.

    Two behaviours matter here beyond plain recursion:

    - **Temporal arithmetic.** ``$origin + $offset::minutes`` must not push a NUMERIC expectation
      onto ``$origin``. When the expectation is temporal, each operand keeps its own shape: a
      cast-to-duration stays numeric underneath, everything else stays temporal.
    - **Literal seeding.** ``$icd_version == "9"`` tells us ``"9"`` is a value the config actually
      branches on. Feeding it back into the column's value pool is what makes the generated data
      exercise both sides of the branch instead of always taking one.

    Args:
        node: The node to visit.
        expect: The expectation on this node's result.
        ctx: The resolving context.
        cs: The accumulator.
        note: Provenance text.
        seen: Derived names already being expanded.
        range_hint: Numeric bounds implied by the surrounding expression.
    """
    key = getattr(node, "KEY", "")
    children = _children(node)

    if key in (
        "equal",
        "not_equal",
        "greater_than",
        "less_than",
        "greater_than_or_equal",
        "less_than_or_equal",
    ):
        _seed_comparison(children, ctx, cs, note, seen)
        child_expect = ValueKind.UNKNOWN
    elif key in ("and", "or", "not"):
        child_expect = ValueKind.BOOLEAN
    elif key in ("add", "subtract") and (range_hint or expect is ValueKind.DATETIME_STR):
        # Integer arithmetic whose result is read as a date: the only coherent reading is a year on
        # the left and an offset (an age, in every real config that does this) on the right. Both
        # MIMIC-IV's `($anchor_year - $anchor_age)::str ... ::year` and NWICU's `... ::?"%Y"` land
        # here. Without the magnitude hint the default 1-100 range makes `year - age` a nonsense
        # year and every birth event is dropped.
        years = range_hint or YEAR_RANGE
        for i, child in enumerate(children):
            sub_hint = years if i == 0 else YEAR_OFFSET_RANGE
            # INTEGER, not NUMERIC: a float year renders as "1883.966", which neither `::year` nor
            # a "%Y" parse accepts. Real anchor-year/age columns are integers anyway.
            _visit(child, ValueKind.INTEGER, ctx, cs, note, seen, sub_hint)
        return
    elif key in ("add", "subtract") and expect in (ValueKind.DATETIME, ValueKind.DATE):
        for child in children:
            sub = ValueKind.NUMERIC if _is_duration_cast(child) else expect
            _visit(child, sub, ctx, cs, note, seen)
        return
    elif key in ("add", "subtract", "multiply", "divide", "power", "mean", "min", "max", "negate"):
        child_expect = ValueKind.NUMERIC
    else:
        child_expect = expect

    for child in children:
        _visit(child, child_expect, ctx, cs, note, seen)


def _seed_comparison(
    children: list[NodeBase],
    ctx: _Context,
    cs: ConstraintSet,
    note: str,
    seen: frozenset[str],
) -> None:
    """Feed a comparison's literal operand back into the other operand's value pool.

    Args:
        children: The comparison's operands.
        ctx: The resolving context.
        cs: The accumulator.
        note: Provenance text.
        seen: Derived names already being expanded.
    """
    literals = [_literal_value(c) for c in children if isinstance(c, Literal)]
    present = [v for v in literals if v is not None]
    if not present:
        return
    values = tuple(str(v) for v in present if not isinstance(v, bool))
    # The literal's Python type tells us the column's type. `$admissioncount == 1` compares against
    # an integer, so generating a string token there makes polars raise "cannot compare string with
    # numeric type" at extraction — a hard failure that no amount of value-pool seeding fixes.
    kind = _kind_of_literal(present[0])
    for child in children:
        if isinstance(child, Literal):
            continue
        _decorate(
            child,
            ctx,
            cs,
            ColumnConstraint(kind=kind, required_values=values).with_note(
                f"{note}: compared to {values or present}"
            ),
            seen,
        )


def _kind_of_literal(value: object) -> ValueKind:
    """Map a Python literal to the column kind it implies when compared against.

    Args:
        value: The literal's value.

    Returns:
        The implied :class:`~MESSY_synth.constraints.ValueKind`.

    Examples:
        >>> [_kind_of_literal(v).name for v in (True, 1, 1.5, "x")]
        ['BOOLEAN', 'INTEGER', 'NUMERIC', 'CATEGORICAL']
    """
    if isinstance(value, bool):
        return ValueKind.BOOLEAN
    if isinstance(value, int):
        return ValueKind.INTEGER
    if isinstance(value, float):
        return ValueKind.NUMERIC
    return ValueKind.CATEGORICAL


def _decorate(
    node: NodeBase,
    ctx: _Context,
    cs: ConstraintSet,
    constraint: ColumnConstraint,
    seen: frozenset[str],
) -> None:
    """Attach extra evidence to whatever column(s) ``node`` bottoms out in.

    Used for facts that belong to a *column* rather than to the expectation flowing down — a
    strptime format, a regex, a minimum length, a compared-against literal. Plain column references
    and derived names are followed; anything more complex is left alone, since the evidence would
    not apply cleanly to a computed value.

    Args:
        node: The node whose underlying column should be decorated.
        ctx: The resolving context.
        cs: The accumulator.
        constraint: The evidence to record.
        seen: Derived names already being expanded.
    """
    match node:
        case Column():
            name = node.args[0]
            if name in ctx.derived:
                if name not in seen:
                    _decorate(ctx.derived[name], ctx, cs, constraint, seen | {name})
                return
            cs.observe(ctx.owner_of(name), name, constraint)
        case Coalesce():
            # Every branch of a `??` chain is a candidate value for the same slot, so evidence
            # about that slot applies to all of them. This is what carries a strptime format onto
            # both columns of `dod_final: $deathtime ?? $dod`.
            for arg in node.args:
                _decorate(arg, ctx, cs, constraint, seen)
        case Conditional():
            for branch in ("then", "otherwise"):
                if (b := node.kwargs.get(branch)) is not None:
                    _decorate(b, ctx, cs, constraint, seen)
        case Strptime() | Cast() | SetTime():
            # Single-source wrappers are transparent for the purpose of column-level evidence: a
            # `MEDS_DEATH` timestamp is still that column's role whether or not it is parsed on the
            # way in. `SetTime` keeps its source in args rather than kwargs.
            source = node.kwargs.get("source") or (node.args[0] if node.args else None)
            if isinstance(source, NodeBase):
                _decorate(source, ctx, cs, constraint, seen)
        case _:
            return


def _children(node: NodeBase) -> list[NodeBase]:
    """Return every child node of ``node``, from both positional args and keyword args.

    Args:
        node: The node to inspect.

    Returns:
        The child nodes, positional first.

    Examples:
        >>> from dftly import Parser
        >>> [type(c).__name__ for c in _children(Parser()("$a + $b"))]
        ['Column', 'Column']
        >>> [type(c).__name__ for c in _children(Parser()('$a::"%Y"'))]
        ['Literal', 'Column']
    """
    out = [a for a in getattr(node, "args", ()) if isinstance(a, NodeBase)]
    out.extend(v for v in (getattr(node, "kwargs", {}) or {}).values() if isinstance(v, NodeBase))
    return out


def _is_duration_cast(node: NodeBase) -> bool:
    """Return whether ``node`` casts to a duration unit.

    Args:
        node: The node to test.

    Returns:
        True if this is a ``::minutes``-style cast.

    Examples:
        >>> from dftly import Parser
        >>> _is_duration_cast(Parser()("$x::minutes")), _is_duration_cast(Parser()("$x"))
        (True, False)
    """
    if not isinstance(node, Cast):
        return False
    return _literal_value(node.kwargs.get("type")) in _NUMERIC_SOURCE_CASTS


def _literal_value(node: object) -> object:
    """Unwrap a dftly ``Literal`` to its Python value, passing anything else through as None.

    Args:
        node: The node (or None) to unwrap.

    Returns:
        The literal's value, or None.

    Examples:
        >>> from dftly import Parser
        >>> _literal_value(Parser()("'MAIN'"))
        'MAIN'
        >>> _literal_value(Parser()("$x")) is None
        True
        >>> _literal_value(None) is None
        True
    """
    return node.args[0] if isinstance(node, Literal) and node.args else None

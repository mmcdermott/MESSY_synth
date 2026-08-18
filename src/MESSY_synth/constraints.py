"""The value-kind lattice used to describe what a synthetic source column must contain.

A MESSY config never states the schema of its own inputs. It states *expressions over* those
inputs, in the `dftly <https://github.com/mmcdermott/dftly>`_ DSL. This module defines the
vocabulary that :mod:`MESSY_synth.inference` uses to write down what it learns by walking those
expressions, plus the rules for combining two independent observations about the same column.

The central type is :class:`ColumnConstraint`. It is a *lattice element*: constraints combine with
:meth:`ColumnConstraint.merge`, which is associative and commutative, so the order in which a
config's expressions happen to be visited never changes the inferred schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum


class ValueKind(IntEnum):
    """What a column must hold, ordered by how tightly it pins the on-disk bytes.

    The integer values *are* the merge precedence: when one column is observed twice with two
    different kinds, the numerically larger kind wins. The ordering runs from "we learned nothing"
    up to "the exact characters are determined".

    The ordering is deliberate, not incidental:

    - :attr:`DATETIME_STR` outranks everything because a ``strptime`` format determines the literal
      characters in the file. A column read as ``$x::"%Y-%m-%d"`` *and* interpolated into an
      f-string is still a ``%Y-%m-%d`` string; the f-string merely stringifies it.
    - :attr:`DATETIME` and :attr:`DATE` outrank the numeric kinds, since a column used directly
      where a timestamp is expected must carry a real temporal dtype.
    - :attr:`NUMERIC` outranks :attr:`CATEGORICAL`: rendering a number into an f-string is fine,
      but feeding a category name into arithmetic is not.
    - :attr:`SUBJECT_ID` sits above the plain numeric kinds because MEDS-Extract casts subject ids
      to ``Int64`` with ``strict=True``; that is a stronger promise than "some number".

    Examples:
        >>> ValueKind.DATETIME_STR > ValueKind.NUMERIC > ValueKind.CATEGORICAL
        True
        >>> max(ValueKind.CATEGORICAL, ValueKind.STRING)
        <ValueKind.CATEGORICAL: 20>
    """

    UNKNOWN = 0
    STRING = 10
    CATEGORICAL = 20
    BOOLEAN = 30
    IDENTIFIER = 40
    NUMERIC = 50
    INTEGER = 60
    SUBJECT_ID = 70
    DATE = 80
    DATETIME = 90
    DATETIME_STR = 100


#: Kinds whose synthetic values are drawn from a small, reusable vocabulary rather than sampled
#: independently per row. Categorical columns are what event ``code`` expressions interpolate, and
#: what ``_metadata`` dictionaries join against, so their value pool has to be finite and shared.
VOCABULARY_KINDS = frozenset({ValueKind.CATEGORICAL, ValueKind.IDENTIFIER})

#: Kinds that must be written as a temporal dtype (or a string parsed into one).
TEMPORAL_KINDS = frozenset({ValueKind.DATE, ValueKind.DATETIME, ValueKind.DATETIME_STR})

#: Kinds that must be written as a number.
NUMERIC_KINDS = frozenset({ValueKind.NUMERIC, ValueKind.INTEGER, ValueKind.SUBJECT_ID})


@dataclass(frozen=True)
class ColumnConstraint:
    """Everything inference has learned about one source column.

    Attributes:
        kind: The strongest :class:`ValueKind` observed for this column.
        datetime_formats: ``strptime`` format strings this column is parsed with, in first-seen
            order. More than one is normal and meaningful: the 0.7 idiom for a column with mixed
            formats is ``coalesce($x::?"%Y-%m-%d %H:%M:%S", $x::?"%Y-%m-%d")``, and a generator
            that emits only the first format would leave the second branch of that coalesce
            untested. Generation distributes rows across every recorded format.
        required_values: Literal values this column is compared against somewhere in the config
            (e.g. the ``"9"`` in ``$icd_version == "9"``). Seeding these into the value pool is
            what makes a config's conditional branches actually branch, instead of every row
            taking the same path.
        match_patterns: Regexes the column is *tested* against with ``/re/ in $col``. Generation
            emits a mix of matching and non-matching values, so both sides of the resulting
            conditional are exercised.
        extract_patterns: Regexes the column has a value *pulled out of* with
            ``extract group N of /re/ from $col``. Unlike a test, a failed extract yields null and
            usually poisons everything downstream, so every generated value must match these.
        numeric_range: Inferred bounds for a numeric column, where the surrounding arithmetic
            implies a magnitude the value kind alone cannot express — the ``$anchor_year -
            $anchor_age`` idiom, whose difference has to render as a four-digit year.
        observed_kinds: Every kind ever recorded for this column, not just the winning one. Kept
            so contradictory usage (a column read both as a formatted string *and* as a native
            timestamp) can be reported rather than silently resolved.
        min_chars: Minimum character length, accumulated from substring slices (``$col[0:3]``
            needs at least 3 characters) and from ``len_chars($col) > n`` comparisons.
        nullable: Whether the column may contain nulls. Set False for columns whose nullity would
            break the run outright — subject ids and join keys.
        plain_time: True when this column supplies the timestamp of an *ordinary* (non-birth,
            non-death) event. It suppresses :attr:`temporal_role`: a column doing double duty
            cannot be placed on the timeline as a death date, because a death date is null for
            every subject who does not die, which would empty the ordinary event too.
        subject_key: True when this column is the *input* to a ``hash()``/``signed_hash()`` that
            produces the subject id. Such a column is not itself a subject id — it can be any text
            — but it stands in one-to-one for a subject, so its pool must be sized by the requested
            subject count and shared across every table that hashes it.
        temporal_role: ``"birth"`` or ``"death"`` when this column supplies the timestamp of a
            ``MEDS_BIRTH`` / ``MEDS_DEATH`` event. MEDS treats those two as the bounds of a
            subject's timeline, and many downstream tools assume birth precedes every other event,
            so generation places them deliberately rather than sampling them like any other date.
        strict_parse: True when at least one ``strptime`` over this column is strict (``::
            rather than ``::?``). A strict parse drops nothing silently; an unparsable value is a
            hard error, so strictly-parsed columns never receive nulls or off-format values.
        notes: Human-readable provenance, one line per observation. Surfaced by the CLI's
            ``--explain`` output so a user can see *why* a column was typed the way it was.

    Examples:
        >>> ColumnConstraint(kind=ValueKind.CATEGORICAL).kind.name
        'CATEGORICAL'

        Merging keeps the stronger kind and unions the evidence:

        >>> a = ColumnConstraint(kind=ValueKind.CATEGORICAL, required_values=("9",))
        >>> b = ColumnConstraint(kind=ValueKind.DATETIME_STR, datetime_formats=("%Y-%m-%d",))
        >>> merged = a.merge(b)
        >>> merged.kind.name, merged.datetime_formats, merged.required_values
        ('DATETIME_STR', ('%Y-%m-%d',), ('9',))

        Merging is commutative, so visit order cannot change the result:

        >>> a.merge(b) == b.merge(a)
        True

        Nullability is conjunctive — one usage that forbids nulls forbids them everywhere:

        >>> ColumnConstraint(nullable=True).merge(ColumnConstraint(nullable=False)).nullable
        False

        A temporal role survives being merged with a role-less observation of the same column:

        >>> ColumnConstraint().merge(ColumnConstraint(temporal_role="birth")).temporal_role
        'birth'

        Merging remembers every kind it saw, not only the winner, so contradictory usage stays
        visible:

        >>> a = ColumnConstraint(kind=ValueKind.DATETIME)
        >>> b = ColumnConstraint(kind=ValueKind.DATETIME_STR)
        >>> sorted(k.name for k in a.merge(b).observed_kinds)
        ['DATETIME', 'DATETIME_STR']
    """

    kind: ValueKind = ValueKind.UNKNOWN
    datetime_formats: tuple[str, ...] = ()
    required_values: tuple[str, ...] = ()
    match_patterns: tuple[str, ...] = ()
    extract_patterns: tuple[str, ...] = ()
    numeric_range: tuple[float, float] | None = None
    observed_kinds: frozenset[ValueKind] = frozenset()
    min_chars: int = 0
    nullable: bool = True
    strict_parse: bool = False
    plain_time: bool = False
    subject_key: bool = False
    temporal_role: str | None = None
    notes: tuple[str, ...] = ()

    def merge(self, other: ColumnConstraint) -> ColumnConstraint:
        """Combine two independent observations of the same column.

        Args:
            other: The constraint to fold in.

        Returns:
            A new constraint holding the stronger kind and the union of all evidence.

        Examples:
            >>> x = ColumnConstraint(kind=ValueKind.NUMERIC, min_chars=2)
            >>> y = ColumnConstraint(kind=ValueKind.STRING, min_chars=5, strict_parse=True)
            >>> z = x.merge(y)
            >>> z.kind.name, z.min_chars, z.strict_parse
            ('NUMERIC', 5, True)

            Duplicate evidence is de-duplicated while preserving first-seen order:

            >>> p = ColumnConstraint(datetime_formats=("%Y", "%Y-%m"))
            >>> q = ColumnConstraint(datetime_formats=("%Y-%m", "%Y-%m-%d"))
            >>> p.merge(q).datetime_formats
            ('%Y', '%Y-%m', '%Y-%m-%d')
        """
        return ColumnConstraint(
            kind=max(self.kind, other.kind),
            datetime_formats=_ordered_union(self.datetime_formats, other.datetime_formats),
            required_values=_ordered_union(self.required_values, other.required_values),
            match_patterns=_ordered_union(self.match_patterns, other.match_patterns),
            extract_patterns=_ordered_union(self.extract_patterns, other.extract_patterns),
            numeric_range=self.numeric_range or other.numeric_range,
            observed_kinds=(self.observed_kinds | other.observed_kinds | {self.kind, other.kind})
            - {ValueKind.UNKNOWN},
            min_chars=max(self.min_chars, other.min_chars),
            nullable=self.nullable and other.nullable,
            strict_parse=self.strict_parse or other.strict_parse,
            plain_time=self.plain_time or other.plain_time,
            subject_key=self.subject_key or other.subject_key,
            temporal_role=self.temporal_role or other.temporal_role,
            notes=_ordered_union(self.notes, other.notes),
        )

    @property
    def effective_temporal_role(self) -> str | None:
        """The birth/death role to actually generate by, or None.

        A role is dropped when the same column also supplies an ordinary event's timestamp. HiRID
        does exactly this: ``date_of_death: '$datetime if $_died'`` bottoms out in the same
        ``datetime`` column that its main observation table reads. Honouring the role there would
        fill that column from each subject's death date — null for everyone who survives — and
        silently delete most of the largest table in the dataset.

        Returns:
            ``"birth"``, ``"death"``, or None.

        Examples:
            >>> ColumnConstraint(temporal_role="death").effective_temporal_role
            'death'
            >>> ColumnConstraint(temporal_role="death", plain_time=True).effective_temporal_role is None
            True
        """
        return None if self.plain_time else self.temporal_role

    @property
    def effective_nullable(self) -> bool:
        """Whether generation may emit nulls for this column.

        A death timestamp is nullable whatever the parse strictness says: most subjects do not die
        inside the record, and polars' ``strptime`` passes nulls straight through rather than
        failing on them. Treating a strict parse as forbidding nulls would force every synthetic
        subject to have a death date.

        Returns:
            True if nulls are permitted.

        Examples:
            >>> ColumnConstraint(nullable=False).effective_nullable
            False
            >>> ColumnConstraint(nullable=False, temporal_role="death").effective_nullable
            True

            Unless the role was suppressed because the column also times an ordinary event:

            >>> ColumnConstraint(nullable=False, temporal_role="death", plain_time=True).effective_nullable
            False
        """
        return self.nullable or self.effective_temporal_role == "death"

    def with_note(self, note: str) -> ColumnConstraint:
        """Return a copy carrying one more provenance line.

        Args:
            note: The line to append.

        Returns:
            The updated constraint.

        Examples:
            >>> ColumnConstraint().with_note("used as time").notes
            ('used as time',)
        """
        return replace(self, notes=_ordered_union(self.notes, (note,)))


def _ordered_union(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Concatenate tuples, dropping duplicates and preserving first-seen order.

    ``dict.fromkeys`` is the idiomatic order-preserving de-duplication in Python 3.7+.

    Args:
        *groups: The tuples to combine.

    Returns:
        The de-duplicated concatenation.

    Examples:
        >>> _ordered_union(("a", "b"), ("b", "c"), ())
        ('a', 'b', 'c')
        >>> _ordered_union()
        ()
    """
    return tuple(dict.fromkeys(item for group in groups for item in group))


@dataclass
class ConstraintSet:
    """A mutable accumulator mapping ``(table_prefix, column)`` to its merged constraint.

    Inference walks every expression in a MESSY config and calls :meth:`observe` each time it
    reaches a column reference. Because :meth:`ColumnConstraint.merge` is commutative, the walk may
    visit expressions in any order.

    Examples:
        >>> cs = ConstraintSet()
        >>> cs.observe("labs", "value", ColumnConstraint(kind=ValueKind.NUMERIC))
        >>> cs.observe("labs", "value", ColumnConstraint(kind=ValueKind.CATEGORICAL))
        >>> cs.get("labs", "value").kind.name
        'NUMERIC'

        Unobserved columns come back as the bottom element rather than raising:

        >>> cs.get("labs", "never_seen").kind.name
        'UNKNOWN'
        >>> sorted(cs.columns_for("labs"))
        ['value']
    """

    constraints: dict[tuple[str, str], ColumnConstraint] = field(default_factory=dict)

    def observe(self, prefix: str, column: str, constraint: ColumnConstraint) -> None:
        """Record one observation about ``prefix``/``column``.

        Args:
            prefix: The source-table prefix the column belongs to.
            column: The column name.
            constraint: What was learned.
        """
        key = (prefix, column)
        existing = self.constraints.get(key)
        self.constraints[key] = constraint if existing is None else existing.merge(constraint)

    def get(self, prefix: str, column: str) -> ColumnConstraint:
        """Look up the merged constraint for a column.

        Args:
            prefix: The source-table prefix.
            column: The column name.

        Returns:
            The merged constraint, or an empty one if the column was never observed.
        """
        return self.constraints.get((prefix, column), ColumnConstraint())

    def columns_for(self, prefix: str) -> set[str]:
        """Return every column observed for one table prefix.

        Args:
            prefix: The source-table prefix.

        Returns:
            The set of observed column names.
        """
        return {col for (p, col) in self.constraints if p == prefix}

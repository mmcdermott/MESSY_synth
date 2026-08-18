"""Value factories: turn a column's inferred constraint into concrete synthetic data.

Every value this module produces is meant to be *unmistakably* synthetic. The goal of the package
is to reproduce a dataset's **shape** — file names, column names, dtypes, string formats, key
relationships — never its content. Categorical values are therefore emitted as ``SYNTH_``-prefixed
tokens rather than plausible clinical vocabulary, timestamps live in an obviously invented window,
and free text is a numbered placeholder. Nothing here is derived from, or resembles, any real
source dataset.

That styling is also load-bearing for correctness, not just for honesty. MEDS-Extract infers dtypes
when it reads CSV sources, but reads ``_metadata`` dictionary files with ``infer_schema=False`` so
their keys keep their exact text. A numeric-looking key like ``01.90`` would be inferred as the
float ``1.9`` on the event side while staying the string ``"01.90"`` on the dictionary side, and
the metadata join would silently match nothing. Because every categorical pool is guaranteed to
contain at least one alphabetic ``SYNTH_`` token (see :func:`categorical_pool`), polars always
infers such a column as a string and the two sides keep agreeing.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from random import Random
from typing import TYPE_CHECKING

from .regexgen import generate_match, generate_non_match

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .constraints import ColumnConstraint

#: Prefix stamped on every generated categorical/identifier token, so synthetic data is obvious at
#: a glance in any downstream artifact.
SYNTH_PREFIX = "SYNTH"

# Naive datetimes throughout: MEDS timestamps are timezone-naive by specification, so attaching a
# tzinfo here would produce values the schema rejects.
#: Earliest possible synthetic birth date.
BIRTH_START = datetime(1930, 1, 1)  # noqa: DTZ001
#: Latest possible synthetic birth date.
BIRTH_END = datetime(1995, 1, 1)  # noqa: DTZ001
#: Start of the window in which timed events are placed.
EVENT_START = datetime(2010, 1, 1)  # noqa: DTZ001
#: Length, in days, of each subject's personal event window.
SUBJECT_WINDOW_DAYS = 1095


@dataclass(frozen=True)
class SubjectTimeline:
    """The invented life history of one synthetic subject.

    Timestamps are not sampled independently. MEDS treats ``MEDS_BIRTH`` and ``MEDS_DEATH`` as the
    bounds of a subject's record, and downstream tooling routinely assumes birth precedes every
    other event, so each subject gets a coherent timeline up front and every timestamp generated
    for that subject is drawn from inside it.

    Attributes:
        subject_id: The subject's integer id.
        birth: The subject's birth timestamp; always before :attr:`window_start`.
        window_start: Start of this subject's event window.
        window_end: End of this subject's event window.
        death: The subject's death timestamp, after :attr:`window_end`, or None if they survive.
    """

    subject_id: int
    birth: datetime
    window_start: datetime
    window_end: datetime
    death: datetime | None

    def sample(self, rng: random.Random) -> datetime:
        """Draw one timestamp from inside this subject's event window.

        Args:
            rng: The random source.

        Returns:
            A timestamp between :attr:`window_start` and :attr:`window_end`.

        Examples:
            >>> tl = build_timeline(1, random.Random(0))
            >>> t = tl.sample(random.Random(1))
            >>> tl.window_start <= t <= tl.window_end
            True
            >>> tl.birth < tl.window_start
            True
        """
        span = int((self.window_end - self.window_start).total_seconds())
        return self.window_start + timedelta(seconds=rng.randint(0, max(span, 1)))


def build_timeline(subject_id: int, rng: random.Random, death_probability: float = 0.3) -> SubjectTimeline:
    """Invent a coherent timeline for one subject.

    Args:
        subject_id: The subject's integer id.
        rng: The random source.
        death_probability: Fraction of subjects that get a death timestamp.

    Returns:
        The subject's :class:`SubjectTimeline`.

    Examples:
        >>> tl = build_timeline(7, random.Random(7))
        >>> tl.subject_id
        7

        The invariant that makes the data usable downstream — birth first, then the event window,
        then death — holds by construction:

        >>> tl.birth < tl.window_start < tl.window_end
        True
        >>> tl.death is None or tl.death > tl.window_end
        True

        Timelines are a deterministic function of the seed, so reruns reproduce byte-identically:

        >>> build_timeline(7, random.Random(7)) == build_timeline(7, random.Random(7))
        True
    """
    birth_span = int((BIRTH_END - BIRTH_START).total_seconds())
    birth = BIRTH_START + timedelta(seconds=rng.randint(0, birth_span))
    window_start = EVENT_START + timedelta(days=rng.randint(0, 365))
    window_end = window_start + timedelta(days=SUBJECT_WINDOW_DAYS)
    death = None
    if rng.random() < death_probability:
        death = window_end + timedelta(days=rng.randint(1, 365))
    return SubjectTimeline(
        subject_id=subject_id,
        birth=birth,
        window_start=window_start,
        window_end=window_end,
        death=death,
    )


@dataclass
class ValueFactory:
    """Generates values for one column, honoring its inferred constraint.

    Attributes:
        rng: The random source. Seeded per column so a column's values depend only on the config,
            the master seed, and the column's identity — never on generation order.
        null_fraction: Fraction of nullable columns' values that come back as None.
        numeric_ranges: Per-column-name numeric bounds, overriding the default range. Useful where
            a config does arithmetic that implies a magnitude inference cannot see — NWICU's
            ``anchor_year``, whose difference from ``anchor_age`` must format as a ``%Y`` year, is
            the canonical example.
    """

    rng: random.Random
    null_fraction: float = 0.05
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def token(self, column: str, index: int, min_chars: int = 0) -> str:
        """Build one synthetic categorical token.

        Args:
            column: The column the token belongs to, used to make tokens self-describing.
            index: The token's index within its pool.
            min_chars: Pad the result to at least this length, for columns the config slices.

        Returns:
            A token such as ``SYNTH_ITEMID_003``.

        Examples:
            >>> ValueFactory(random.Random(0)).token("itemid", 3)
            'SYNTH_ITEMID_003'

            Columns the config slices (``$icd_code[0:4]``) are padded so the slice is well-defined:

            >>> ValueFactory(random.Random(0)).token("x", 1, min_chars=20)
            'SYNTH_X_001XXXXXXXXX'
        """
        slug = "".join(ch if ch.isalnum() else "_" for ch in column).upper().strip("_") or "COL"
        base = f"{SYNTH_PREFIX}_{slug}_{index:03d}"
        if len(base) < min_chars:
            base += "X" * (min_chars - len(base))
        return base

    def numeric(
        self,
        column: str,
        integral: bool = False,
        inferred_range: tuple[float, float] | None = None,
    ) -> float | int:
        """Draw one number.

        Precedence is explicit override, then inferred range, then the default. A user who passes
        ``-R col=lo:hi`` is correcting the inference, so their bound has to win.

        Args:
            column: The column name, used to look up a range override.
            integral: Whether to return an int rather than a float.
            inferred_range: Bounds deduced from the surrounding arithmetic, if any.

        Returns:
            The generated number.

        Examples:
            >>> f = ValueFactory(random.Random(0), numeric_ranges={"anchor_year": (1950, 2000)})
            >>> 1950 <= f.numeric("anchor_year", integral=True) <= 2000
            True
            >>> isinstance(f.numeric("value"), float)
            True

            An inferred range applies where no override exists, and yields to one where it does:

            >>> 1900 <= f.numeric("other", integral=True, inferred_range=(1900, 2000)) <= 2000
            True
            >>> 1950 <= f.numeric("anchor_year", True, inferred_range=(1, 2)) <= 2000
            True
        """
        lo, hi = self.numeric_ranges.get(column) or inferred_range or (1.0, 100.0)
        if integral:
            return self.rng.randint(int(lo), int(hi))
        return round(self.rng.uniform(lo, hi), 3)

    def text(self, column: str) -> str:
        """Draw one free-text value.

        Args:
            column: The column name, echoed into the value so output is self-describing.

        Returns:
            A placeholder string.

        Examples:
            >>> ValueFactory(random.Random(0)).text("note")
            'SYNTH_NOTE_TEXT_...'
        """
        slug = "".join(ch if ch.isalnum() else "_" for ch in column).upper().strip("_") or "COL"
        return f"{SYNTH_PREFIX}_{slug}_TEXT_{self.rng.randint(0, 999):03d}"

    def maybe_null(self, value: object, nullable: bool) -> object:
        """Replace a value with None at the configured rate.

        Nulls are worth injecting deliberately: the 0.7 ``??`` coalescing idiom and the null-drop
        semantics of composite codes only get exercised when some values are actually missing.

        Args:
            value: The candidate value.
            nullable: Whether this column tolerates nulls at all.

        Returns:
            Either ``value`` or None.

        Examples:
            >>> f = ValueFactory(random.Random(0), null_fraction=0.0)
            >>> f.maybe_null("x", nullable=True)
            'x'
            >>> f2 = ValueFactory(random.Random(0), null_fraction=1.0)
            >>> f2.maybe_null("x", nullable=True) is None
            True

            A column that cannot tolerate nulls never receives one, whatever the rate:

            >>> f2.maybe_null("x", nullable=False)
            'x'
        """
        if not nullable:
            return value
        return None if self.rng.random() < self.null_fraction else value


def categorical_pool(column: str, constraint: ColumnConstraint, size: int, rng: Random) -> list[str]:
    """Build the shared vocabulary for one categorical or identifier pool.

    Four rules make the pool useful rather than merely well-typed:

    - **Literal seeding.** Any value the config compares this column against
      (:attr:`~MESSY_synth.constraints.ColumnConstraint.required_values`, e.g. the ``"9"`` in
      ``$icd_version == "9"``) goes into the pool, so the config's conditional branches actually
      take both paths instead of every row falling through the same way.
    - **Extract patterns are mandatory.** Where the config pulls a value *out* of this column
      (``extract /2003|2010/ from $admissionyeargroup``), every member must match, because a failed
      extract nulls the derived column and usually deletes every event in the table.
    - **Match patterns are mixed.** Where a regex merely *tests* the column (``/^E/ in $icd_code``),
      roughly half the pool matches and half does not, so both branches get exercised.
    - **Guaranteed alphabetic member.** Absent any pattern, at least one ``SYNTH_`` token is present,
      which forces polars to infer the column as a string. Without it an all-numeric pool would be
      read as a number on the event side while a ``_metadata`` dictionary kept it as text, and the
      metadata join would match nothing.

    Args:
        column: The column name, used to make tokens self-describing.
        constraint: The merged constraint for the pool.
        size: How many distinct values to produce.
        rng: The random source.

    Returns:
        The pool's values.

    Examples:
        >>> c = ColumnConstraint(kind=ValueKind.CATEGORICAL)
        >>> categorical_pool("itemid", c, 3, random.Random(0))
        ['SYNTH_ITEMID_000', 'SYNTH_ITEMID_001', 'SYNTH_ITEMID_002']

        Compared-against literals are seeded first, and still leave an alphabetic token behind:

        >>> seeded = ColumnConstraint(kind=ValueKind.CATEGORICAL, required_values=("9", "10"))
        >>> categorical_pool("icd_version", seeded, 3, random.Random(0))
        ['9', '10', 'SYNTH_ICD_VERSION_000']

        A pool of size 1 that must also carry a required literal grows rather than dropping the
        alphabetic guarantee, since losing it would break metadata joins:

        >>> categorical_pool("v", ColumnConstraint(required_values=("9",)), 1, random.Random(0))
        ['9', 'SYNTH_V_000']

        Slice constraints propagate into the token width:

        >>> categorical_pool("icd", ColumnConstraint(min_chars=14), 1, random.Random(0))
        ['SYNTH_ICD_000X']

        An extract pattern makes every member conform:

        >>> yeargroup = ColumnConstraint(extract_patterns=("2003|2010",))
        >>> pool = categorical_pool("admissionyeargroup", yeargroup, 4, random.Random(0))
        >>> all(("2003" in v) or ("2010" in v) for v in pool)
        True

        A test pattern instead produces a mix, so the conditional it gates goes both ways:

        >>> tested = ColumnConstraint(match_patterns=("^E",))
        >>> pool = categorical_pool("icd_code", tested, 6, random.Random(0))
        >>> matches = [v.startswith("E") for v in pool]
        >>> any(matches) and not all(matches)
        True
    """
    values: list[str] = list(dict.fromkeys(constraint.required_values))
    factory = ValueFactory(Random(0))
    needed = max(1, size - len(values))

    if constraint.extract_patterns:
        values.extend(_pattern_values(constraint.extract_patterns, needed, rng, match=True))
    elif constraint.match_patterns:
        # Half matching, half not; at least one of each where the size allows.
        n_match = max(1, needed // 2)
        values.extend(_pattern_values(constraint.match_patterns, n_match, rng, match=True))
        values.extend(_pattern_values(constraint.match_patterns, needed - n_match, rng, match=False))
    else:
        values.extend(factory.token(column, i, constraint.min_chars) for i in range(needed))

    deduped = list(dict.fromkeys(values))
    if not deduped:
        deduped = [factory.token(column, 0, constraint.min_chars)]
    return deduped


def _pattern_values(patterns: tuple[str, ...], count: int, rng: Random, *, match: bool) -> list[str]:
    r"""Generate ``count`` values that all match (or all avoid) every pattern.

    Args:
        patterns: The regexes to satisfy or avoid.
        count: How many values to produce.
        rng: The random source.
        match: True to satisfy the patterns, False to avoid them.

    Returns:
        The generated values, possibly fewer than ``count`` if generation failed.

    Examples:
        >>> _pattern_values((r"^\d{2}$",), 2, random.Random(1), match=True)
        ['41', '63']
        >>> _pattern_values(("^E",), 1, random.Random(1), match=False)
        ['NOMATCH000']
    """
    out: list[str] = []
    for _ in range(count * 4):
        if len(out) >= count:
            break
        head, *rest = patterns
        candidate = generate_match(head, rng) if match else generate_non_match(head, rng)
        if candidate is None:
            break
        # Multiple patterns on one column must all hold at once.
        ok = all(bool(re.search(p, candidate)) is match for p in rest)
        if ok and candidate not in out:
            out.append(candidate)
    return out


def format_datetime(value: datetime, constraint: ColumnConstraint, row_index: int) -> str:
    """Render a timestamp using one of the column's recorded ``strptime`` formats.

    When a column carries several formats — the 0.7 idiom for mixed-format columns is
    ``coalesce($x::?"%Y-%m-%d %H:%M:%S", $x::?"%Y-%m-%d")`` — rows are distributed across all of
    them, so every branch of the coalesce is exercised rather than just the first.

    Args:
        value: The timestamp to render.
        constraint: The column's constraint, supplying the format list.
        row_index: The row's position, used to rotate through the formats.

    Returns:
        The rendered string.

    Examples:
        >>> dt = datetime(2015, 6, 1, 13, 45, 0)  # noqa: DTZ001
        >>> c = ColumnConstraint(datetime_formats=("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"))
        >>> format_datetime(dt, c, 0)
        '2015-06-01 13:45:00'
        >>> format_datetime(dt, c, 1)
        '2015-06-01'

        With no recorded format, ISO-8601 is the fallback:

        >>> format_datetime(dt, ColumnConstraint(), 0)
        '2015-06-01T13:45:00'
    """
    formats = constraint.datetime_formats
    if not formats:
        return value.isoformat()
    return value.strftime(formats[row_index % len(formats)])

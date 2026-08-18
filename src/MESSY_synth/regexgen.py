"""Generate strings that satisfy a regex, so regex-driven expressions actually produce values.

Real MESSY configs route load-bearing logic through regexes. AmsterdamUMCdb derives every
timestamp in its ``admissions`` table from::

    _origin: '(((extract /2003|2010/ from $admissionyeargroup)::int)::year)::datetime'

If the synthetic ``admissionyeargroup`` does not contain ``2003`` or ``2010``, the extract yields
null, ``_origin`` is null, and *every* event in the table silently disappears. NWICU does the same
with ``$icd_code[0:4] ... if /^E/ in $icd_code``. Generating opaque tokens for these columns
produces a structurally valid dataset that extracts to nothing.

So this module reverses a pattern into a conforming string. It implements the subset of regex
syntax that appears in practice — literals, character classes, escapes, alternation, groups,
quantifiers, and anchors — and *verifies* every result with :func:`re.search` before returning it.
Anything it cannot handle returns None rather than a wrong answer, and the caller falls back to an
ordinary synthetic token.
"""

from __future__ import annotations

import re
import string
from random import Random  # noqa: TC003 - used at runtime by this module's doctests

#: Characters used to satisfy ``\w``, ``\d``, ``\s`` and friends. Deliberately narrow: the point is
#: a value that matches, not one that explores the whole space.
_CLASS_CHARS: dict[str, str] = {
    "d": string.digits,
    "w": string.ascii_uppercase + string.digits + "_",
    "s": " ",
    "D": string.ascii_uppercase,
    "W": " -",
    "S": string.ascii_uppercase + string.digits,
}

#: Fallback alphabet for ``.`` and for negated classes.
_DEFAULT_ALPHABET = string.ascii_uppercase + string.digits

#: How many repetitions an unbounded quantifier (``*``, ``+``) produces.
_UNBOUNDED_REPEAT = 3


class _UnsupportedPatternError(Exception):
    """Raised internally when a pattern uses syntax this generator does not model."""


def generate_match(pattern: str, rng: Random, attempts: int = 12) -> str | None:
    r"""Return a string matching ``pattern``, or None if one cannot be produced.

    The result is always checked with :func:`re.search` before it is returned, so a caller can
    trust a non-None result even for patterns whose parse is approximate.

    Args:
        pattern: The regex, in Python/Rust-compatible syntax.
        rng: The random source.
        attempts: How many candidates to try before giving up.

    Returns:
        A matching string, or None.

    Examples:
        >>> rng = random.Random(0)
        >>> generate_match("2003|2010", rng)
        '2010'

        Character classes and quantifiers are honored, which is what makes a banded value like
        AmsterdamUMCdb's ``agegroup`` come out usable:

        >>> agegroup = generate_match(r"^(\d{2})", rng)
        >>> agegroup.isdigit() and len(agegroup) == 2
        True

        A quantified *group* repeats the whole group, literals included — resampling only the last
        character class would silently drop the ``AB``:

        >>> generate_match(r"(AB\d){2}", random.Random(0))
        'AB6AB4'

        Anchors contribute no characters:

        >>> generate_match("^E", rng)
        'E'

        Literal text passes through unchanged:

        >>> generate_match(r"ICD\-10", rng)
        'ICD-10'

        A pattern this generator cannot model returns None rather than a wrong answer, so the
        caller can fall back:

        >>> generate_match("(?<=x)y", rng) is None
        True
    """
    try:
        compiled = re.compile(pattern)
    except re.error:
        return None

    for _ in range(attempts):
        try:
            candidate = _Generator(pattern, rng).run()
        except _UnsupportedPatternError:
            return None
        if compiled.search(candidate):
            return candidate
    return None


def generate_non_match(pattern: str, rng: Random, attempts: int = 12) -> str | None:
    """Return a string that does *not* match ``pattern``.

    Used so a config's regex conditionals take both branches. A synthetic dataset where
    ``/^E/ in $icd_code`` is always true never exercises the else-branch, and a bug there would go
    unnoticed.

    Args:
        pattern: The regex to avoid.
        rng: The random source.
        attempts: How many candidates to try.

    Returns:
        A non-matching string, or None if none was found.

    Examples:
        >>> rng = random.Random(0)
        >>> value = generate_non_match("^E", rng)
        >>> value.startswith("E")
        False
    """
    try:
        compiled = re.compile(pattern)
    except re.error:
        return None
    for i in range(attempts):
        candidate = f"NOMATCH{rng.randint(0, 999):03d}" if i else "NOMATCH000"
        if not compiled.search(candidate):
            return candidate
    return None


class _Generator:
    """A recursive-descent regex reader that emits one conforming string as it parses.

    Parsing and generation are fused: there is no AST, because nothing here needs to inspect the
    pattern twice. Each parse method returns the text it consumed's contribution to the output.
    """

    def __init__(self, pattern: str, rng: Random) -> None:
        """Initialize the generator.

        Args:
            pattern: The regex source.
            rng: The random source.
        """
        self.pattern = pattern
        self.pos = 0
        self.rng = rng

    def run(self) -> str:
        """Generate one string for the whole pattern.

        Returns:
            The generated string.

        Raises:
            _UnsupportedPatternError: If trailing input remains unconsumed.
        """
        out = self._alternation()
        if self.pos < len(self.pattern):
            raise _UnsupportedPatternError(f"unconsumed input at {self.pos}")
        return out

    def _peek(self) -> str | None:
        """Return the next character without consuming it.

        Returns:
            The character, or None at end of input.
        """
        return self.pattern[self.pos] if self.pos < len(self.pattern) else None

    def _alternation(self) -> str:
        """Parse ``a|b|c`` and generate one randomly chosen branch.

        Returns:
            The generated text.
        """
        branches = [self._concat()]
        while self._peek() == "|":
            self.pos += 1
            branches.append(self._concat())
        return self.rng.choice(branches)

    def _concat(self) -> str:
        """Parse a sequence of quantified atoms.

        Returns:
            The concatenated generated text.
        """
        parts: list[str] = []
        while (ch := self._peek()) is not None and ch not in "|)":
            parts.append(self._quantified())
        return "".join(parts)

    def _quantified(self) -> str:
        r"""Parse one atom plus any quantifier that follows it.

        Each repetition re-parses the atom from its original position rather than resampling a
        remembered alphabet. That distinction matters for a *composite* atom: resampling the last
        character class seen would turn ``(AB\d){2}`` into two bare digits, dropping the ``AB``
        entirely, while re-parsing reproduces the whole group. It also keeps ``\d{4}`` from coming
        out as ``1111``.

        Returns:
            The generated text, repeated as the quantifier requires.
        """
        start = self.pos
        first = self._atom()
        low, high = self._quantifier()
        after = self.pos
        if low == 1 and high == 1:
            return first
        count = self.rng.randint(low, high)
        if count == 0:
            return ""
        parts = [first]
        for _ in range(count - 1):
            self.pos = start
            parts.append(self._atom())
        self.pos = after
        return "".join(parts)

    def _quantifier(self) -> tuple[int, int]:
        """Parse a quantifier suffix, if present.

        Returns:
            The (min, max) repetition bounds.

        Raises:
            _UnsupportedPatternError: On a malformed ``{n,m}``.
        """
        ch = self._peek()
        if ch == "*":
            self.pos += 1
            bounds = (0, _UNBOUNDED_REPEAT)
        elif ch == "+":
            self.pos += 1
            bounds = (1, _UNBOUNDED_REPEAT)
        elif ch == "?":
            self.pos += 1
            bounds = (0, 1)
        elif ch == "{":
            close = self.pattern.find("}", self.pos)
            if close == -1:
                raise _UnsupportedPatternError("unterminated {")
            body = self.pattern[self.pos + 1 : close]
            self.pos = close + 1
            try:
                if "," in body:
                    lo_s, _, hi_s = body.partition(",")
                    lo = int(lo_s) if lo_s else 0
                    hi = int(hi_s) if hi_s else int(lo) + _UNBOUNDED_REPEAT
                else:
                    lo = hi = int(body)
            except ValueError as e:
                raise _UnsupportedPatternError(f"bad repeat {{{body}}}") from e
            bounds = (lo, hi)
        else:
            return (1, 1)
        # A trailing `?` only makes the quantifier lazy; it does not change what can match.
        if self._peek() == "?":
            self.pos += 1
        return bounds

    def _atom(self) -> str:
        """Parse a single atom and generate its text.

        Returns:
            The generated text.

        Raises:
            _UnsupportedPatternError: On lookarounds, backreferences, or other unmodeled syntax.
        """
        ch = self._peek()
        if ch is None:
            raise _UnsupportedPatternError("unexpected end of pattern")

        if ch in "^$":
            # Anchors constrain position, not content.
            self.pos += 1
            return ""
        if ch == "(":
            return self._group()
        if ch == "[":
            return self._char_class()
        if ch == "\\":
            return self._escape()
        if ch == ".":
            self.pos += 1
            return self.rng.choice(_DEFAULT_ALPHABET)
        self.pos += 1
        return ch

    def _group(self) -> str:
        """Parse ``(...)``, ``(?:...)``, and named groups.

        Returns:
            The generated text for the group's body.

        Raises:
            _UnsupportedPatternError: On lookaround or other extension syntax.
        """
        self.pos += 1  # consume "("
        if self.pattern.startswith("?", self.pos):
            if self.pattern.startswith("?:", self.pos):
                self.pos += 2
            elif self.pattern.startswith("?P<", self.pos):
                close = self.pattern.find(">", self.pos)
                if close == -1:
                    raise _UnsupportedPatternError("unterminated group name")
                self.pos = close + 1
            else:
                # Lookahead/lookbehind/inline flags change matching semantics in ways a
                # left-to-right generator cannot honor.
                raise _UnsupportedPatternError("lookaround or inline flags")
        body = self._alternation()
        if self._peek() != ")":
            raise _UnsupportedPatternError("unterminated group")
        self.pos += 1
        return body

    def _char_class(self) -> str:
        """Parse ``[...]`` including ranges and negation.

        Returns:
            One character from the class.

        Raises:
            _UnsupportedPatternError: On an unterminated or empty class.
        """
        self.pos += 1  # consume "["
        negated = self._peek() == "^"
        if negated:
            self.pos += 1

        members: list[str] = []
        while (ch := self._peek()) is not None and ch != "]":
            if ch == "\\":
                members.extend(self._escape_alphabet())
                continue
            self.pos += 1
            if self._peek() == "-" and self.pos + 1 < len(self.pattern) and self.pattern[self.pos + 1] != "]":
                self.pos += 1
                end = self.pattern[self.pos]
                self.pos += 1
                members.extend(chr(c) for c in range(ord(ch), ord(end) + 1))
            else:
                members.append(ch)

        if self._peek() != "]":
            raise _UnsupportedPatternError("unterminated character class")
        self.pos += 1

        alphabet = "".join(dict.fromkeys(members))
        if negated:
            alphabet = "".join(c for c in _DEFAULT_ALPHABET if c not in alphabet)
        if not alphabet:
            raise _UnsupportedPatternError("empty character class")
        return self.rng.choice(alphabet)

    def _escape(self) -> str:
        """Parse a backslash escape and generate one character.

        Returns:
            The generated character.
        """
        alphabet = self._escape_alphabet()
        return self.rng.choice(alphabet)

    def _escape_alphabet(self) -> str:
        """Consume a backslash escape and return the alphabet it denotes.

        Returns:
            The characters the escape can match.

        Raises:
            _UnsupportedPatternError: On a dangling backslash or a backreference.
        """
        self.pos += 1  # consume "\"
        ch = self._peek()
        if ch is None:
            raise _UnsupportedPatternError("dangling backslash")
        self.pos += 1
        if ch in _CLASS_CHARS:
            return _CLASS_CHARS[ch]
        if ch.isdigit():
            raise _UnsupportedPatternError("backreference")
        # An escaped metacharacter (\., \-, \/) stands for itself.
        return ch

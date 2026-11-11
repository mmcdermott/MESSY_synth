"""Edge cases of generating values that satisfy several regex patterns at once."""

from __future__ import annotations

import random

from MESSY_synth import values


def test_a_pattern_that_stops_generating_after_working_drops_out(monkeypatch):
    """A pattern that produced a valid value and later cannot generate is dropped, not retried."""
    calls: list[str] = []

    def fake_generate_match(pattern: str, rng: random.Random) -> str | None:
        calls.append(pattern)
        # "^A" yields one value that also satisfies "B$", then fails; "B$" never generates.
        if pattern == "^A" and calls.count("^A") == 1:
            return "AB"
        return None

    monkeypatch.setattr(values, "generate_match", fake_generate_match)
    out = values._pattern_values(("^A", "B$"), 3, random.Random(0), match=True)

    assert out == ["AB"]
    assert calls == ["^A", "^A", "B$"]

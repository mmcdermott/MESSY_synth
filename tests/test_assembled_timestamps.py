"""Timestamps a config assembles from several source columns are generated without registrations.

The config never states each column's format, so none is inferred. Instead a whole timestamp is rendered with
the format the config parses, and cut back into the pieces the expression reassembles.
"""

import polars as pl
import pytest
from MEDS_extract.config import MessyConfig

from MESSY_synth import GenerationOptions, generate, infer_constraints, synthesize


def config(*times, **columns):
    """Build a config with one timed event per ``time`` expression."""
    events = {f"e{i}": {"code": f"X{i}", "time": time, **columns} for i, time in enumerate(times)}
    return MessyConfig.parse(
        {
            "etl": {"dataset_name": "Assembled", "raw_dataset_version": "1"},
            "t": {"_defaults": {"subject_id": "$pid"}, **events},
        }
    )


def parses(frame: pl.DataFrame, expr: pl.Expr, fmt: str) -> bool:
    """Return whether every non-null assembled value parses under ``fmt``."""
    assembled = frame.select(expr.alias("s"))["s"].drop_nulls()
    return assembled.len() > 0 and assembled.str.to_datetime(fmt, strict=False).null_count() == 0


def test_split_date_and_clock_need_no_registration(tmp_path):
    """The CDSL shape: a sliced date column and a clock column joined by a space."""
    fmt = "%Y-%m-%d %H:%M:%S"
    cfg = config(f'f"{{$date[0:10]}} {{$clock}}"::?"{fmt}"')
    result = synthesize(cfg, tmp_path, null_fraction=0.0)
    assert result.ok, [str(f) for f in result.findings]
    frame = result.dataset.frames["t"]
    assert frame["date"].str.len_chars().to_list() == [10] * frame.height
    assert parses(frame, pl.col("date").str.slice(0, 10) + " " + pl.col("clock"), fmt)


def test_no_column_format_is_inferred():
    """Generation satisfies the expression; inference still claims nothing about either column."""
    cs = infer_constraints(config('f"{$date[0:10]} {$clock}"::?"%Y-%m-%d %H:%M:%S"'))
    assert cs.get("t", "date").datetime_formats == ()
    assert cs.get("t", "clock").datetime_formats == ()


def test_a_column_shared_by_several_assembled_timestamps_satisfies_all(tmp_path):
    """One date column, two clock columns with different precision, as in CDSL's emergency events."""
    cfg = config(
        'f"{$date[0:10]} {$first}"::?"%Y-%m-%d %H:%M"',
        'f"{$date[0:10]} {$last}"::?"%Y-%m-%d %H:%M:%S"',
    )
    result = synthesize(cfg, tmp_path, null_fraction=0.0)
    assert result.ok, [str(f) for f in result.findings]
    frame = result.dataset.frames["t"]
    assert parses(frame, pl.col("date") + " " + pl.col("first"), "%Y-%m-%d %H:%M")
    assert parses(frame, pl.col("date") + " " + pl.col("last"), "%Y-%m-%d %H:%M:%S")


def test_three_pieces_with_a_repeated_separator(tmp_path):
    """No format splitting: the separator also occurs inside the format."""
    result = synthesize(config('f"{$y}-{$m}-{$d}"::"%Y-%m-%d"'), tmp_path, null_fraction=0.0)
    assert result.ok, [str(f) for f in result.findings]
    frame = result.dataset.frames["t"]
    assert parses(frame, pl.concat_str("y", "m", "d", separator="-"), "%Y-%m-%d")


def test_a_nonzero_slice_is_padded_rather_than_given_a_format(tmp_path):
    """``$raw[3:13]`` parses as a date, so the slice is a date; the rest of the column is filler."""
    cfg = config('$raw[3:13]::"%Y-%m-%d"')
    assert infer_constraints(cfg).get("t", "raw").datetime_formats == ()
    result = synthesize(cfg, tmp_path, null_fraction=0.0)
    assert result.ok, [str(f) for f in result.findings]
    assert parses(result.dataset.frames["t"], pl.col("raw").str.slice(3, 10), "%Y-%m-%d")


def test_one_column_sliced_twice_in_one_template(tmp_path):
    """Both slices of the same column must come from one consistent value."""
    result = synthesize(config('f"{$s[0:4]}/{$s[5:7]}"::"%Y/%m"'), tmp_path, null_fraction=0.0)
    assert result.ok, [str(f) for f in result.findings]
    frame = result.dataset.frames["t"]
    assert parses(frame, pl.col("s").str.slice(0, 4) + "/" + pl.col("s").str.slice(5, 2), "%Y/%m")


def test_coalesced_alternatives_are_not_conflicts(tmp_path):
    """Only one branch of a coalesce has to parse, so the others are not reported."""
    cfg = config('coalesce(f"{$a} {$b}"::?"%Y-%m-%d %H:%M", f"{$a} {$b}"::?"%Y-%m-%d %H:%M:%S")')
    result = synthesize(cfg, tmp_path, null_fraction=0.0)
    assert result.ok, [str(f) for f in result.findings]
    assert not result.dataset.assembly_conflicts
    assert not [f for f in result.findings if f.level != "info"]


def test_registered_pieces_are_kept_and_the_rest_fit_around_them(tmp_path):
    """A user registration is fixed; the unregistered piece is cut to match it."""
    from MESSY_synth import chrono_formatter

    cfg = config('f"{$date} {$clock}"::?"%Y-%m-%d %H:%M:%S"')
    result = synthesize(
        cfg,
        tmp_path,
        null_fraction=0.0,
        column_generators={("t", "date"): chrono_formatter("%Y-%m-%d")},
    )
    assert result.ok, [str(f) for f in result.findings]
    assert parses(result.dataset.frames["t"], pl.col("date") + " " + pl.col("clock"), "%Y-%m-%d %H:%M:%S")


def test_an_unsatisfiable_assembly_is_reported_by_column():
    """A fixed piece that cannot fit is named, rather than surfacing only as an empty event."""
    cfg = config('f"{$date} {$clock}"::?"%Y-%m-%d %H:%M:%S"')
    dataset = generate(
        cfg,
        GenerationOptions(null_fraction=0.0, column_generators={("t", "date"): lambda value: "nope"}),
    )
    assert dataset.assembly_conflicts
    (assembly, rows), *_ = dataset.assembly_conflicts.items()
    assert assembly.columns == ("date", "clock")
    assert rows == dataset.frames["t"].height


def test_generation_stays_deterministic():
    """Assembled values come from seeded instants."""
    cfg = config('f"{$date[0:10]} {$clock}"::?"%Y-%m-%d %H:%M:%S"')
    assert generate(cfg).frames["t"].equals(generate(cfg).frames["t"])


@pytest.mark.parametrize("template,fmt", [("{$a} {$b}", "%d %b %Y %H:%M"), ("{$a}{$b}", "%Y%m%d%H%M%S")])
def test_formats_with_spaces_or_no_separators(tmp_path, template, fmt):
    """The cut follows the template's literals, whatever the format looks like."""
    result = synthesize(config(f'f"{template}"::"{fmt}"'), tmp_path, null_fraction=0.0)
    assert result.ok, [str(f) for f in result.findings]


def test_a_template_the_format_can_never_produce_is_reported(tmp_path):
    """A literal space the format never renders cannot be cut; say so, naming the columns."""
    result = synthesize(config('f"{$a} {$b}"::"%Y%m%d%H%M%S"'), tmp_path, null_fraction=0.0)
    assert not result.ok
    assert any(f.level == "warning" and "columns a, b" in f.where for f in result.findings)


def test_a_numeric_piece_stays_numeric(tmp_path):
    """A piece the config also compares numerically keeps its numeric values, so that event still works."""
    cfg = MessyConfig.parse(
        {
            "etl": {"dataset_name": "Assembled", "raw_dataset_version": "1"},
            "t": {
                "_defaults": {"subject_id": "$pid"},
                "stamp": {"code": "X", "time": 'f"{$d} {$hr}:{$mi}"::"%Y-%m-%d %H:%M"'},
                "big_hr": {"code": '"BIG" if $hr > 50 else "SMALL"', "time": "$ts"},
            },
        }
    )
    result = synthesize(cfg, tmp_path, seed=0, null_fraction=0.0)
    big_hr = result.events.filter(event="big_hr").row(0, named=True)
    assert big_hr["error"] is None, big_hr["error"]
    assert big_hr["rows_out"] > 0
    assert any(f.level == "warning" and "hr (read as INTEGER)" in str(f) for f in result.findings)


def test_a_required_literal_survives_assembly(tmp_path):
    """A literal the config compares a piece to keeps its rows, or the conflict is reported."""
    cfg = MessyConfig.parse(
        {
            "etl": {"dataset_name": "Assembled", "raw_dataset_version": "1"},
            "t": {
                "_defaults": {"subject_id": "$pid"},
                "stamp": {"code": "X", "time": 'f"{$d} {$tag}"::"%Y-%m-%d %H:%M"'},
                "special": {"code": '"SPECIAL" if $tag == "AM" else "OTHER"', "time": "$ts"},
            },
        }
    )
    result = synthesize(cfg, tmp_path, seed=0, null_fraction=0.0)
    n_am = (result.dataset.frames["t"]["tag"] == "AM").sum()
    assert n_am > 0 or not result.ok, "SPECIAL branch can never fire, yet synthesize() reports ok"
    assert any(f.level == "warning" and "tag (compared to 'AM')" in str(f) for f in result.findings)

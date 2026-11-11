"""Regression checks for explicit generation without reverse-engineering expressions."""

from datetime import datetime

import polars as pl
import pytest
from MEDS_extract.config import MessyConfig

from MESSY_synth import (
    GenerationOptions,
    chrono_formatter,
    generate,
    infer_constraints,
    smoke_test,
    synthesize,
)


def config(time, **columns):
    """Build a minimal timed-event config."""
    return MessyConfig.parse(
        {
            "etl": {"dataset_name": "Generators", "raw_dataset_version": "1"},
            "t": {"_defaults": {"subject_id": "$pid"}, "e": {"code": "X", "time": time, **columns}},
        }
    )


@pytest.mark.parametrize("fraction", ["%.f", "%.3f", "%.6f", "%.9f"])
def test_chrono_formats_round_trip_through_polars(fraction):
    """Formatting and parsing use the same directive language, including precision."""
    fmt = "%Y-%m-%d %H:%M:%S" + fraction
    value = datetime(2015, 6, 1, 13, 45, 0, 123456)
    rendered = chrono_formatter(fmt)(value)
    parsed = pl.Series([rendered]).str.to_datetime(fmt).item()
    assert parsed == value.replace(microsecond=123000 if fraction == "%.3f" else 123456)


def test_chrono_escaped_percent_and_invalid_directive():
    """Literal percent signs are not rewritten and unknown directives fail visibly."""
    value = datetime(2015, 6, 1)
    assert chrono_formatter("%%.f")(value) == "%.f"
    with pytest.raises(pl.exceptions.ComputeError):
        chrono_formatter("%Q")(value)


def test_slices_do_not_declare_a_whole_column_format(tmp_path):
    """A nonzero slice can parse a timestamp without requiring the source to be one."""
    cfg = config('$raw[3:13]::"%Y-%m-%d"')
    assert infer_constraints(cfg).get("t", "raw").datetime_formats == ()
    result = synthesize(
        cfg,
        tmp_path,
        null_fraction=0,
        column_generators={("t", "raw"): lambda value: "TS:" + value.strftime("%Y-%m-%d")},
    )
    assert result.ok, result.findings


def test_registered_columns_share_row_instant_and_respect_temporal_roles():
    """Registration must not silently replace births/deaths with ordinary event times."""
    cfg = config('f"{$date} {$clock}"::?"%Y-%m-%d %H:%M:%S"', text_value="$whole")
    options = GenerationOptions(
        null_fraction=0,
        column_generators={
            ("t", "date"): chrono_formatter("%Y-%m-%d"),
            ("t", "clock"): chrono_formatter("%H:%M:%S"),
            ("t", "whole"): chrono_formatter("%Y-%m-%d %H:%M:%S"),
        },
    )
    ds = generate(cfg, options)
    frame = ds.frames["t"]
    assert (frame["date"] + " " + frame["clock"]).equals(frame["whole"])
    assert generate(cfg, options).frames["t"].equals(frame)

    cfg = MessyConfig.parse(
        {
            "etl": {"dataset_name": "Roles", "raw_dataset_version": "1"},
            "t": {
                "_defaults": {"subject_id": "$pid"},
                "birth": {"code": "MEDS_BIRTH", "time": "$birth"},
                "death": {"code": "MEDS_DEATH", "time": "$death"},
            },
        }
    )
    ds = generate(
        cfg,
        GenerationOptions(
            null_fraction=0,
            column_generators={("t", "birth"): lambda value: value, ("t", "death"): lambda value: value},
        ),
    )
    for row in ds.frames["t"].iter_rows(named=True):
        assert row["birth"] == ds.timelines[row["pid"]].birth
        assert row["death"] == ds.timelines[row["pid"]].death


def test_bad_registered_values_are_checked_by_messy_expressions(tmp_path):
    """Both time and non-time expressions participate in dry-run validation."""
    bad_time = synthesize(
        config('$ts::?"%Y-%m-%d"'),
        tmp_path / "time",
        column_generators={("t", "ts"): lambda value: "not-a-date"},
    )
    assert not bad_time.ok
    bad_numeric = synthesize(
        config('$ts::"%Y-%m-%d"', numeric_value="$value::float64"),
        tmp_path / "numeric",
        column_generators={("t", "value"): lambda value: "not-a-number"},
    )
    assert not bad_numeric.ok
    assert any("failed to evaluate" in finding.message for finding in bad_numeric.findings)


@pytest.mark.parametrize("column,generator,error", [("pid", str, ValueError), ("ts", 42, TypeError)])
def test_unsafe_registrations_fail_early(column, generator, error):
    """Subject keys cannot be overwritten and strategies must be callable."""
    with pytest.raises(error):
        generate(config('$ts::"%Y-%m-%d"'), GenerationOptions(column_generators={("t", column): generator}))


def test_generator_cannot_return_null_for_non_nullable_column():
    """Generated non-null constraints apply even when the callback returns the null itself."""
    with pytest.raises(ValueError, match="non-nullable column at row 0"):
        generate(
            config('$ts::"%Y-%m-%d"'),
            GenerationOptions(column_generators={("t", "ts"): lambda value: None}),
        )


def test_smoke_test_accepts_registered_generators(tmp_path):
    """The public end-to-end entry point forwards explicit generators to synthesis."""
    import yaml

    path = tmp_path / "messy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "etl": {"dataset_name": "Registered", "raw_dataset_version": "1"},
                "t": {
                    "_defaults": {"subject_id": "$pid"},
                    "e": {"code": "X", "time": '$raw[3:13]::"%Y-%m-%d"'},
                },
            }
        )
    )
    result = smoke_test(
        path,
        tmp_path / "run",
        column_generators={("t", "raw"): lambda value: "TS:" + value.strftime("%Y-%m-%d")},
    )
    assert result.ok, result.summary()


def test_cli_registers_chrono_formats_for_split_timestamps(tmp_path):
    """CI runs the CLI, so split date/clock columns must be registrable without Python."""
    import yaml

    from MESSY_synth.__main__ import main

    path = tmp_path / "messy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "etl": {"dataset_name": "Split", "raw_dataset_version": "1"},
                "t": {
                    "_defaults": {"subject_id": "$pid"},
                    "e": {"code": "X", "time": 'f"{$date} {$clock}"::?"%Y-%m-%d %H:%M:%S"'},
                },
            }
        )
    )
    assert main([str(path), "-o", str(tmp_path / "bare"), "-q"]) == 1
    registered = ["--column-format", "t/date=%Y-%m-%d", "--column-format", "t/clock=%H:%M:%S"]
    assert main([str(path), "-o", str(tmp_path / "registered"), "-q", *registered]) == 0


def verdicts(result):
    """Return the per-registration findings, keyed by column."""
    return {f.where: f for f in result.findings if "registration" in f.message}


def test_each_registration_gets_a_matches_verdict(tmp_path):
    """A registration that the config's expressions accept is reported as matching, by name."""
    result = synthesize(
        config('f"{$date} {$clock}"::?"%Y-%m-%d %H:%M:%S"'),
        tmp_path,
        column_generators={
            ("t", "date"): chrono_formatter("%Y-%m-%d"),
            ("t", "clock"): chrono_formatter("%H:%M:%S"),
        },
    )
    assert result.ok, result.findings
    found = verdicts(result)
    assert set(found) == {"t:date", "t:clock"}
    assert all(f.level == "info" and "matches" in f.message and "t/e" in f.message for f in found.values())


def test_a_registration_the_config_rejects_gets_a_does_not_match_verdict(tmp_path):
    """The verdict names the registered column and the event that rejected its values."""
    result = synthesize(
        config('$ts::"%Y-%m-%d"', numeric_value="$value::float64"),
        tmp_path,
        column_generators={
            ("t", "ts"): chrono_formatter("%Y-%m-%d"),
            ("t", "value"): lambda value: "not-a-number",
        },
    )
    assert not result.ok
    found = verdicts(result)
    assert found["t:ts"].level == "info"
    assert found["t:value"].level == "error"
    assert "does not match" in found["t:value"].message and "t/e" in found["t:value"].message


def test_no_verdicts_without_validation(tmp_path):
    """Verdicts come from the dry run, so skipping validation skips them."""
    result = synthesize(
        config('$ts::"%Y-%m-%d"'),
        tmp_path,
        validate=False,
        column_generators={("t", "ts"): chrono_formatter("%Y-%m-%d")},
    )
    assert not verdicts(result)

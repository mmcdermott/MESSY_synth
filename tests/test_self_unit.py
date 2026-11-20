"""A ``_self`` ``unit`` gives every code one unit, as MEDS-Extract 0.9 requires.

MEDS-Extract 0.9 writes a code's ``unit`` metadata as one String and fails the run when the rows of
one code carry distinct units (0.8 collected them into a list). A ``_self`` block reads the unit
off the event's own rows, so a unit column drawn independently per row gives most codes several
units — a dataset the real ETL rejects, although the config is valid for data where each code is
measured in one unit. The generator therefore makes such a unit column a function of the code.
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml
from MEDS_extract.config import MessyConfig

from MESSY_synth import GenerationOptions, generate
from MESSY_synth.smoke import smoke_test

BASE = {
    "etl": {
        "dataset_name": "SelfUnit",
        "raw_dataset_version": "1",
        "n_subjects_per_shard": 100,
        "split_fracs": {"train": 0.5, "tuning": 0.25, "held_out": 0.25},
    },
    "_defaults": {"subject_id": "$pid"},
    "patients": {"dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'}},
}

OPTIONS = GenerationOptions(seed=0, n_subjects=40, rows_per_subject=6)


def _config(labs: dict, **tables: dict) -> dict:
    return {**BASE, "labs": labs, **tables}


def _units_per_group(frame: pl.DataFrame, keys: list[str], unit: str) -> int:
    """The most distinct non-null units any one ``keys`` group carries."""
    return frame.drop_nulls(keys).group_by(keys).agg(pl.col(unit).drop_nulls().n_unique())[unit].max()


def test_a_self_unit_outside_the_code_is_one_unit_per_code():
    """A unit column the code does not read gets one value per code, nulls kept."""
    cfg = MessyConfig.parse(
        _config(
            {
                "lab": {
                    "code": 'f"LAB//{$itemid}"',
                    "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
                    "numeric_value": "$valuenum",
                    "_metadata": {"_self": {"unit": "$valueuom"}},
                }
            }
        )
    )
    labs = generate(cfg, OPTIONS).frames["labs"]
    assert _units_per_group(labs, ["itemid"], "valueuom") == 1
    # The unit's own nulls are kept where they were drawn, and the unit still varies across codes.
    assert labs["valueuom"].null_count() > 0
    assert labs["valueuom"].n_unique() > 2


def test_a_code_built_through_a_join_groups_by_the_joined_label():
    """A code rendered from a joined label has one unit across every key sharing the label."""
    cfg = MessyConfig.parse(
        _config(
            {
                "_table": {"joins": {"items": {"key": "itemid", "cols": ["label"]}}},
                "lab": {
                    "code": 'f"LAB//{$label}"',
                    "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
                    "numeric_value": "$valuenum",
                    "_metadata": {"_self": {"unit": "$valueuom"}},
                },
            }
        )
    )
    frames = generate(cfg, OPTIONS).frames
    joined = frames["labs"].join(frames["items"], on="itemid", how="left")
    assert _units_per_group(joined, ["label"], "valueuom") == 1


def test_a_unit_spelled_in_the_code_is_left_alone():
    """A code that already names its unit has one unit per code; the column keeps its variety."""
    cfg = MessyConfig.parse(
        _config(
            {
                "_table": {"cols": {"units": "$valueuom"}},
                "lab": {
                    "code": "f\"LAB//{$itemid}//{$units ?? 'UNK'}\"",
                    "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
                    "numeric_value": "$valuenum",
                    "_metadata": {"_self": {"unit": "$valueuom"}},
                },
            }
        )
    )
    labs = generate(cfg, OPTIONS).frames["labs"]
    assert _units_per_group(labs, ["itemid"], "valueuom") > 1


@pytest.mark.slow
def test_the_etl_accepts_a_self_unit_outside_the_code(tmp_path):
    """MEDS-Extract 0.9 builds the cohort instead of rejecting several units per code."""
    spec = _config(
        {
            "lab": {
                "code": 'f"LAB//{$itemid}"',
                "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
                "numeric_value": "$valuenum",
                "_metadata": {"_self": {"unit": "$valueuom"}},
            }
        }
    )
    fp = tmp_path / "messy.yaml"
    fp.write_text(yaml.safe_dump(spec, sort_keys=False))
    result = smoke_test(fp, tmp_path / "run", n_subjects=16, rows_per_subject=4, seed=0)
    assert result.ok, result.summary()

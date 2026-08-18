"""Run a full ``meds-extract-run`` over generated data and assert the output is a real dataset.

This is the test that justifies the package. Everything else checks that the generated files *look*
right; this one checks that MEDS-Extract can actually consume them end to end and emit a populated,
schema-valid MEDS cohort.

It is marked ``slow`` because it spawns the real pipeline (eight stages, each its own subprocess).
Deselect with ``-m "not slow"``.
"""

from __future__ import annotations

import polars as pl
import pytest
from MEDS_extract.config import MessyConfig

from MESSY_synth.smoke import smoke_test

#: A small but structurally complete ETL: a subject-level table with birth/death and a static
#: event, a timed event table with a numeric value, a join to a dimension table, and a metadata
#: dictionary keyed on one component of a two-part code.
E2E_CONFIG = {
    "etl": {
        "dataset_name": "SynthDemo",
        "raw_dataset_version": "0.1",
        "n_subjects_per_shard": 100,
        "split_fracs": {"train": 0.5, "tuning": 0.25, "held_out": 0.25},
    },
    "_defaults": {"subject_id": "$pid"},
    "patients": {
        "sex": {"code": "f\"SEX//{$sex ?? 'UNK'}\"", "time": None},
        "dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'},
        "dod": {"code": "MEDS_DEATH", "time": '$dod::?"%Y-%m-%d %H:%M:%S"'},
    },
    "labs": {
        "_table": {"join": {"stays": {"key": "stay_id", "cols": ["pid"]}}},
        "lab": {
            "code": "f\"LAB//{$itemid ?? 'UNK'}//{$uom ?? 'UNK'}\"",
            "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
            "numeric_value": "$valuenum",
            "_metadata": {"d_labitems": {"itemid": "$itemid", "description": "$label"}},
        },
    },
}


@pytest.fixture(scope="module")
def etl_result(tmp_path_factory):
    """Generate synthetic data, run the whole ETL over it, and return the result.

    Args:
        tmp_path_factory: pytest's session-scoped temp directory factory.

    Returns:
        The :class:`~MESSY_synth.smoke.SmokeResult`.
    """
    workdir = tmp_path_factory.mktemp("e2e")
    cfg_fp = workdir / "messy.yaml"
    # Write the spec to disk: `meds-extract-run` resolves a path, not an in-memory object.
    import yaml

    cfg_fp.write_text(yaml.safe_dump(E2E_CONFIG, sort_keys=False))
    return smoke_test(cfg_fp, workdir / "run", n_subjects=16, rows_per_subject=3, seed=0)


@pytest.mark.slow
def test_the_etl_runs_and_produces_a_populated_dataset(etl_result):
    """The pipeline exits 0 and writes a non-empty MEDS cohort."""
    assert etl_result.ok, etl_result.summary()
    assert etl_result.returncode == 0
    assert etl_result.n_events > 0
    assert etl_result.n_subjects == 16


@pytest.mark.slow
def test_output_matches_the_meds_data_schema(etl_result):
    """Shards carry the MEDS columns with the dtypes the schema requires."""
    shards = sorted((etl_result.output_dir / "data").glob("*/*.parquet"))
    assert shards
    frame = pl.concat([pl.read_parquet(p) for p in shards], how="diagonal_relaxed")
    assert {"subject_id", "time", "code"} <= set(frame.columns)
    assert frame["subject_id"].dtype == pl.Int64
    assert frame["time"].dtype == pl.Datetime
    assert frame["code"].dtype == pl.String


@pytest.mark.slow
def test_every_declared_event_reached_the_output(etl_result):
    """Each event in the config contributed rows, so nothing vanished silently."""
    shards = sorted((etl_result.output_dir / "data").glob("*/*.parquet"))
    frame = pl.concat([pl.read_parquet(p) for p in shards], how="diagonal_relaxed")
    blocks = set(frame["source_block"].unique().to_list())
    cfg = MessyConfig.parse(E2E_CONFIG)
    expected = {f"{t.input_prefix}/{e.name}" for t in cfg.event_tables for e in t.events}
    assert expected <= blocks, expected - blocks


@pytest.mark.slow
def test_metadata_descriptions_actually_joined(etl_result):
    """The ``_metadata`` dictionary matched the event's codes rather than silently missing.

    Codes built from a *null* component are excluded. ``f"LAB//{$itemid ?? 'UNK'}//..."`` renders
    the text ``UNK`` when ``itemid`` is null, but MEDS-Extract matches metadata on the component
    value the event stamped — which is null, not ``"UNK"`` — so those codes are unmatchable by
    construction. That is a property of the config idiom, not of the generated data, and real ETLs
    using ``?? 'UNK'`` have it too.
    """
    codes = pl.read_parquet(etl_result.output_dir / "metadata" / "codes.parquet")
    lab_codes = codes.filter(pl.col("code").str.starts_with("LAB//"))
    assert lab_codes.height > 0
    matchable = lab_codes.filter(~pl.col("code").str.contains("//UNK//"))
    assert matchable.height > 0
    assert matchable["description"].is_not_null().all(), matchable.filter(pl.col("description").is_null())


@pytest.mark.slow
def test_births_precede_the_events_they_bound(etl_result):
    """Each subject's MEDS_BIRTH is at or before all of their other timestamps."""
    shards = sorted((etl_result.output_dir / "data").glob("*/*.parquet"))
    frame = pl.concat([pl.read_parquet(p) for p in shards], how="diagonal_relaxed")
    births = frame.filter(pl.col("code") == "MEDS_BIRTH").select("subject_id", birth="time")
    assert births.height > 0
    timed = frame.filter(pl.col("time").is_not_null() & (pl.col("code") != "MEDS_BIRTH"))
    joined = timed.join(births, on="subject_id", how="inner")
    assert (joined["birth"] <= joined["time"]).all()


@pytest.mark.slow
def test_all_splits_are_populated(etl_result):
    """No split rounded to zero subjects, which would make the run useless as a fixture."""
    splits = pl.read_parquet(etl_result.output_dir / "metadata" / "subject_splits.parquet")
    counts = splits.group_by("split").len()
    assert counts.height == 3
    assert counts["len"].min() > 0

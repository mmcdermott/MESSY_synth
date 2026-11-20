"""CSV sources are written with the config's ``etl.csv_separator`` (MEDS-Extract 0.9).

MEDS-Extract 0.9 reads csv sources (event tables, join targets and ``_metadata`` files) with the
field separator ``etl.csv_separator`` names: one character for every source, or a mapping of
source prefix to one under which unlisted prefixes stay ``","``. Comma-separated files fed to such
a config arrive as one column each, so the separator has to be honored on write. MEDS-Extract 0.8
rejects the option, so these tests skip there.
"""

from __future__ import annotations

import pytest
import yaml
from MEDS_extract.config import MessyConfig

from MESSY_synth import synthesize
from MESSY_synth.smoke import smoke_test


def _config(csv_separator: object) -> dict:
    return {
        "etl": {
            "dataset_name": "Separated",
            "raw_dataset_version": "1",
            "n_subjects_per_shard": 100,
            "split_fracs": {"train": 0.5, "tuning": 0.25, "held_out": 0.25},
            "csv_separator": csv_separator,
        },
        "_defaults": {"subject_id": "$pid"},
        "patients": {"dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'}},
        "labs": {
            "_table": {"joins": {"stays": {"key": "stay_id", "cols": ["pid"]}}},
            "lab": {
                "code": 'f"LAB//{$itemid}"',
                "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
                "numeric_value": "$valuenum",
                "_metadata": {"d_labitems": {"itemid": "$itemid", "description": "$label"}},
            },
        },
    }


def _parse(spec: dict) -> MessyConfig:
    try:
        return MessyConfig.parse(spec)
    except ValueError as e:
        if "csv_separator" in str(e):
            pytest.skip("this MEDS-Extract has no etl.csv_separator option (added in 0.9)")
        raise


def _header(path) -> str:
    return path.read_text().splitlines()[0]


def test_every_csv_is_written_with_the_separator(tmp_path):
    result = synthesize(_parse(_config(";")), tmp_path, seed=0, null_fraction=0.0)
    assert result.fmt == "csv"
    assert set(result.files) == {"patients", "labs", "stays", "d_labitems"}
    for path in result.files.values():
        assert ";" in _header(path) and "," not in _header(path), path.name
    # The dry run reads the files as MEDS-Extract does, so every event still yields rows.
    assert result.ok
    assert result.events["error"].null_count() == result.events.height
    assert (result.events["rows_out"] > 0).all()


def test_a_per_prefix_separator_leaves_unlisted_prefixes_comma_separated(tmp_path):
    result = synthesize(_parse(_config({"labs": "\t", "d_labitems": "|"})), tmp_path, seed=0)
    assert "\t" in _header(result.files["labs"])
    assert "|" in _header(result.files["d_labitems"])
    assert "," in _header(result.files["stays"])
    assert result.ok
    assert (result.events["rows_out"] > 0).all()


@pytest.mark.slow
def test_the_etl_reads_the_separated_files(tmp_path):
    spec = _config({"labs": ";", "stays": "\t", "d_labitems": "|"})
    _parse(spec)
    fp = tmp_path / "messy.yaml"
    fp.write_text(yaml.safe_dump(spec, sort_keys=False))
    result = smoke_test(fp, tmp_path / "run", n_subjects=16, rows_per_subject=3, seed=0, fmt="csv")
    assert result.ok, result.summary()
    assert result.n_described_codes > 0

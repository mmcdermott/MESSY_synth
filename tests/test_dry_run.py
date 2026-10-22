"""The dry run evaluates every output expression, not only ``code`` and ``time``."""

import polars as pl
from MEDS_extract.config import MessyConfig

from MESSY_synth import dry_run


def test_a_failing_value_expression_is_an_error(tmp_path):
    """A strict numeric cast that fails is reported, though code and time evaluate fine."""
    cfg = MessyConfig.parse(
        {
            "etl": {"dataset_name": "DryRun", "raw_dataset_version": "1"},
            "t": {
                "_defaults": {"subject_id": "$pid"},
                "e": {"code": "X", "time": '$ts::"%Y-%m-%d"', "numeric_value": "$v::float64"},
            },
        }
    )
    pl.DataFrame({"pid": [1, 2], "ts": ["2015-06-01", "2015-06-02"], "v": ["1.5", "not a number"]}).write_csv(
        tmp_path / "t.csv"
    )
    (row,) = dry_run(cfg, tmp_path).iter_rows(named=True)
    assert row["error"] is not None

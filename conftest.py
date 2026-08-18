"""Test set-up and fixtures code.

The doctest namespace is pre-populated so examples throughout the package read as usage rather
than as boilerplate. `yaml_to_disk` (`yaml_disk`) and `pretty-print-directory` (`print_directory`,
`PrintConfig`) auto-register via their own pytest plugins once installed; everything else is added
here, including the handful of MESSY_synth types that examples construct directly.
"""

import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from MESSY_synth.constraints import ColumnConstraint, ValueKind


@pytest.fixture(scope="session", autouse=True)
def _setup_doctest_namespace(doctest_namespace: dict[str, Any]):
    doctest_namespace.update(
        {
            "datetime": datetime,
            "tempfile": tempfile,
            "Path": Path,
            "pl": pl,
            "random": random,
            "ColumnConstraint": ColumnConstraint,
            "ValueKind": ValueKind,
        }
    )

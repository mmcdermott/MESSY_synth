"""Generate structurally-faithful synthetic raw data from any MEDS-Extract v0.7 MESSY config.

A MESSY config is a complete, config-only ETL: it says where the raw data comes from, and how to
turn each raw table into MEDS events, using the `dftly <https://github.com/mmcdermott/dftly>`_
expression DSL. What it never says is what the raw tables *look like* — those are supposed to
already exist, behind a credentialed download.

MESSY_synth runs that reasoning backwards. It reads the config's expressions, deduces what the
source files must have contained, and writes a directory of synthetic files matching that shape:
right filenames, right columns, right dtypes, right string formats, right keys. The ETL then runs
over them end to end, with no access to the real dataset.

Every generated value is deliberately, visibly fake. The point is to reproduce a dataset's
*structure* so an ETL can be demonstrated and regression-tested — never to imitate its content.

Typical use::

    from MESSY_synth import synthesize

    result = synthesize("path/to/messy.yaml", "raw_synthetic/")
    print(result.events)   # per-event row counts from the dry run
"""

from .constraints import ColumnConstraint, ConstraintSet, ValueKind
from .generate import GeneratedDataset, GenerationOptions, generate
from .inference import infer_constraints
from .plan import ColumnPlan, DatasetPlan, TablePlan, ValuePool, build_plan
from .smoke import SmokeResult, check_output, run_etl, smoke_test
from .synth import SynthesisResult, synthesize
from .validate import Finding, check_plan, dry_run, recommended_n_subjects
from .writer import FORMATS, write_dataset

__all__ = [
    "FORMATS",
    "ColumnConstraint",
    "ColumnPlan",
    "ConstraintSet",
    "DatasetPlan",
    "Finding",
    "GeneratedDataset",
    "GenerationOptions",
    "SmokeResult",
    "SynthesisResult",
    "TablePlan",
    "ValueKind",
    "ValuePool",
    "build_plan",
    "check_output",
    "check_plan",
    "dry_run",
    "generate",
    "infer_constraints",
    "recommended_n_subjects",
    "run_etl",
    "smoke_test",
    "synthesize",
    "write_dataset",
]

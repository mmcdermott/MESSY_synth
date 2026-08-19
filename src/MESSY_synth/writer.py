"""Write generated frames to disk in the layout MEDS-Extract's source resolver accepts.

``MEDS_extract.io.resolve_source_files`` locates a table by *prefix*, trying
``SOURCE_FILE_EXTS = (".parquet", ".par", ".csv.gz", ".csv")`` as a bare file (``labs.csv``) or as a
sub-sharded directory (``labs/*.csv``). A prefix that matches more than one layout — say both
``labs.csv`` and ``labs.parquet`` — is a hard error, so this module writes exactly one file per
prefix. Prefixes containing slashes (``nw_hosp/admissions``) become nested directories.

Choosing the format is not purely cosmetic. MEDS-Extract type-infers CSV sources the way
``pl.read_csv`` does, which does **not** parse date-like strings — they arrive as strings. A config
that uses a column directly as a timestamp (``time: $Offset``, as SICdb does over its pre-MEDS
output) therefore needs a format that carries real dtypes. The default ``"auto"`` detects exactly
that case and writes parquet for the whole dataset, and CSV otherwise, so the common case stays
human-readable while the dtype-dependent case still runs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from .constraints import ValueKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .generate import GeneratedDataset
    from .plan import DatasetPlan

logger = logging.getLogger(__name__)

#: Output formats this module can write. ``auto`` picks between csv and parquet per dataset.
FORMATS = ("auto", "csv", "csv.gz", "parquet")

#: Filename of the provenance manifest dropped alongside the data. It carries no source-file
#: extension, so MEDS-Extract's prefix resolver ignores it.
MANIFEST_NAME = "_MESSY_synth_manifest.json"


def needs_real_dtypes(plan: DatasetPlan) -> bool:
    """Return whether any column must carry a true temporal dtype rather than text.

    A column inferred as :attr:`~MESSY_synth.constraints.ValueKind.DATETIME` or ``DATE`` *without*
    any recorded ``strptime`` format is used as a timestamp directly, so it has to reach
    MEDS-Extract already typed. CSV cannot express that.

    Args:
        plan: The dataset plan.

    Returns:
        True if the dataset must be written in a dtype-preserving format.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> from MESSY_synth.plan import build_plan
        >>> parsed = MessyConfig.parse({
        ...     "t": {"_defaults": {"subject_id": "$pid"},
        ...           "e": {"code": "X", "time": '$ts::"%Y-%m-%d"'}},
        ... })
        >>> needs_real_dtypes(build_plan(parsed))
        False

        The same config reading its timestamp column directly does need them:

        >>> bare = MessyConfig.parse({
        ...     "t": {"_defaults": {"subject_id": "$pid"}, "e": {"code": "X", "time": "$ts"}},
        ... })
        >>> needs_real_dtypes(build_plan(bare))
        True
    """
    return any(
        column.constraint.kind in (ValueKind.DATETIME, ValueKind.DATE)
        and not column.constraint.datetime_formats
        for table in plan.tables
        for column in table.columns
    )


def csv_would_retype(frame: pl.DataFrame) -> bool:
    """Return whether a CSV round-trip would change any of ``frame``'s dtypes.

    MEDS-Extract type-infers CSV sources exactly as ``pl.read_csv`` does, and that inference is
    lossy in both directions that matter here. A datetime column becomes a string, because polars
    does not parse dates unless asked. A *string* column whose values happen to be all digits
    becomes an integer — which is how AmsterdamUMCdb's ``admissionyeargroup`` (legitimately the
    text ``"2003"``) turns into ``2003`` and makes ``extract /2003|2010/ from $admissionyeargroup``
    fail with "expected String type, got: i64".

    Rather than predict those cases from the schema, this does the round trip and compares. It is
    exact, and it needs no heuristic about what "numeric-looking" means.

    Args:
        frame: The frame that would be written.

    Returns:
        True if any column's dtype would change.

    Examples:
        >>> csv_would_retype(pl.DataFrame({"a": ["x", "y"], "n": [1, 2]}))
        False

        A string column of digits does not survive:

        >>> csv_would_retype(pl.DataFrame({"yeargroup": ["2003", "2010"]}))
        True

        Neither does a real timestamp:

        >>> csv_would_retype(pl.DataFrame({"t": [datetime(2020, 1, 1)]}))
        True
    """
    if frame.height == 0:
        return False
    try:
        restored = pl.read_csv(frame.write_csv().encode(), infer_schema_length=None)
    except Exception:  # pragma: no cover - defensive; a frame we cannot round-trip is not csv-safe
        return True
    return dict(restored.schema) != dict(frame.schema)


def resolve_format(
    plan: DatasetPlan,
    fmt: str = "auto",
    frames: dict[str, pl.DataFrame] | None = None,
) -> str:
    """Resolve the ``auto`` format to a concrete one.

    Args:
        plan: The dataset plan.
        fmt: One of :data:`FORMATS`.
        frames: The generated frames. When given, ``auto`` additionally verifies every non-metadata
            frame survives a CSV round trip, catching value-level hazards the plan cannot see.
            Metadata is always read as text by MEDS-Extract.

    Returns:
        ``"csv"``, ``"csv.gz"``, or ``"parquet"``.

    Raises:
        ValueError: If ``fmt`` is not a recognized format.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> from MESSY_synth.plan import build_plan
        >>> plan = build_plan(MessyConfig.parse({
        ...     "t": {"_defaults": {"subject_id": "$pid"},
        ...           "e": {"code": "X", "time": '$ts::"%Y-%m-%d"'}},
        ... }))
        >>> resolve_format(plan)
        'csv'
        >>> resolve_format(plan, "parquet")
        'parquet'

        A frame CSV cannot round-trip forces parquet even when the plan alone looks csv-safe:

        >>> resolve_format(plan, "auto", {"t": pl.DataFrame({"code": ["2003", "2010"]})})
        'parquet'

        >>> resolve_format(plan, "nonsense")
        Traceback (most recent call last):
            ...
        ValueError: Unknown format 'nonsense'. Choose one of: auto, csv, csv.gz, parquet
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format {fmt!r}. Choose one of: {', '.join(FORMATS)}")
    if fmt != "auto":
        return fmt
    if needs_real_dtypes(plan):
        return "parquet"
    if frames and any(
        csv_would_retype(frame) for prefix, frame in frames.items() if not plan.table(prefix).is_metadata
    ):
        return "parquet"
    return "csv"


def write_dataset(
    dataset: GeneratedDataset,
    out_dir: str | Path,
    fmt: str = "auto",
    *,
    write_manifest: bool = True,
) -> dict[str, Path]:
    """Write every generated frame under ``out_dir``.

    Args:
        dataset: The generated dataset.
        out_dir: Directory to write into. Created if absent.
        fmt: One of :data:`FORMATS`.
        write_manifest: Whether to drop a provenance manifest beside the data.

    Returns:
        A mapping of table prefix to the file written.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> from MESSY_synth.generate import generate, GenerationOptions
        >>> cfg = MessyConfig.parse({
        ...     "etl": {"dataset_name": "Demo"},
        ...     "nw_hosp/patients": {
        ...         "_defaults": {"subject_id": "$subject_id"},
        ...         "dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'},
        ...     },
        ... })
        >>> ds = generate(cfg, GenerationOptions(seed=0, n_subjects=3))
        >>> with tempfile.TemporaryDirectory() as d:
        ...     written = write_dataset(ds, d)
        ...     print_directory(Path(d))
        ├── _MESSY_synth_manifest.json
        └── nw_hosp
            └── patients.csv

        The nested prefix became a nested directory, which is exactly what
        ``resolve_source_files`` looks for. The written file round-trips through MEDS-Extract's
        own resolver:

        >>> from MEDS_extract.io import resolve_source_files
        >>> with tempfile.TemporaryDirectory() as d:
        ...     _ = write_dataset(ds, d, write_manifest=False)
        ...     [p.name for p in resolve_source_files(Path(d), "nw_hosp/patients")]
        ['patients.csv']
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_format(dataset.plan, fmt, dataset.frames)

    written: dict[str, Path] = {}
    for table in dataset.plan.tables:
        frame = dataset.frames[table.prefix]
        path = out_dir / f"{table.prefix}.{resolved}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if resolved == "parquet":
            frame.write_parquet(path)
        elif resolved == "csv.gz":
            # polars does not infer compression from the extension — it raises if the name implies
            # one and `compression` does not say so.
            frame.write_csv(path, compression="gzip")
        else:
            frame.write_csv(path)
        written[table.prefix] = path
        logger.info(f"Wrote {frame.height} rows x {frame.width} cols to {path}")

    if write_manifest:
        _write_manifest(dataset, out_dir, resolved, written)
    return written


def _write_manifest(
    dataset: GeneratedDataset,
    out_dir: Path,
    fmt: str,
    written: dict[str, Path],
) -> Path:
    """Record what was generated, so a directory of synthetic data is self-describing.

    Args:
        dataset: The generated dataset.
        out_dir: The output directory.
        fmt: The concrete format written.
        written: The prefix-to-path mapping.

    Returns:
        The manifest path.
    """
    manifest = {
        "generator": "MESSY_synth",
        "warning": "SYNTHETIC DATA. Structure only; every value is invented and meaningless.",
        "dataset_name": dataset.plan.dataset_name,
        "n_subjects": dataset.plan.n_subjects,
        "format": fmt,
        "tables": {
            prefix: {
                "path": str(path.relative_to(out_dir)),
                "rows": dataset.frames[prefix].height,
                "columns": list(dataset.frames[prefix].columns),
            }
            for prefix, path in sorted(written.items())
        },
    }
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path

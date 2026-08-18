"""The one-call entry point: a MESSY config in, a directory of synthetic source data out.

:func:`synthesize` chains the four stages — infer, plan, generate, write — and then validates the
result, so the common case is a single call. The lower-level modules stay available for callers who
want to inspect or override an intermediate step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from MEDS_extract.config import MessyConfig

from .generate import GenerationOptions, generate
from .validate import Finding, check_plan, dry_run, findings_from_dry_run, recommended_n_subjects
from .writer import resolve_format, write_dataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    import polars as pl

    from .generate import GeneratedDataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SynthesisResult:
    """Everything :func:`synthesize` produced.

    Attributes:
        dataset: The generated frames and the plan behind them.
        output_dir: Where the files were written.
        files: Table prefix to written path.
        fmt: The concrete format written.
        findings: Static and dry-run findings, most severe first.
        events: The per-event dry-run table, or None if validation was skipped.
    """

    dataset: GeneratedDataset
    output_dir: Path
    files: dict[str, Path]
    fmt: str
    findings: list[Finding] = field(default_factory=list)
    events: pl.DataFrame | None = None

    @property
    def ok(self) -> bool:
        """Whether the generated dataset is free of errors.

        Returns:
            True if no finding has level ``"error"``.
        """
        return not any(f.level == "error" for f in self.findings)


def synthesize(
    spec: str | Path | MessyConfig,
    output_dir: str | Path,
    *,
    n_subjects: int | None = None,
    rows_per_subject: int = 4,
    vocab_size: int = 8,
    seed: int = 0,
    null_fraction: float = 0.05,
    fmt: str = "auto",
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
    validate: bool = True,
) -> SynthesisResult:
    """Generate and write a synthetic source dataset for a MESSY config.

    Args:
        spec: A registered pipeline name, a ``pkg://`` reference, a path to a MESSY file, or an
            already-parsed :class:`~MEDS_extract.config.MessyConfig`.
        output_dir: Directory to write the source files into.
        n_subjects: Subject-universe size. Defaults to the smallest count that safely populates
            every configured split (see
            :func:`~MESSY_synth.validate.recommended_n_subjects`).
        rows_per_subject: Rows per subject in each timed event table.
        vocab_size: Distinct values per categorical vocabulary.
        seed: Master seed; output is a pure function of (config, options).
        null_fraction: Fraction of nullable values emitted as nulls.
        fmt: One of :data:`~MESSY_synth.writer.FORMATS`.
        numeric_ranges: Per-column-name numeric bounds, overriding the default ``(1, 100)``.
        validate: Whether to run the static checks and the dry run.

    Returns:
        The :class:`SynthesisResult`.

    Examples:
        >>> cfg = MessyConfig.parse({
        ...     "etl": {"dataset_name": "Demo", "raw_dataset_version": "1"},
        ...     "patients": {
        ...         "_defaults": {"subject_id": "$pid"},
        ...         "dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'},
        ...         "sex": {"code": 'f"SEX//{$sex}"', "time": None},
        ...     },
        ...     "labs": {
        ...         "_defaults": {"subject_id": "$pid"},
        ...         "lab": {
        ...             "code": 'f"LAB//{$itemid}"',
        ...             "time": '$ts::"%Y-%m-%d %H:%M:%S"',
        ...             "numeric_value": "$value",
        ...         },
        ...     },
        ... })
        >>> with tempfile.TemporaryDirectory() as d:
        ...     result = synthesize(cfg, d, seed=0, null_fraction=0.0)
        ...     print_directory(Path(d))
        ...     print(result.ok, result.fmt, result.dataset.plan.n_subjects)
        ├── _MESSY_synth_manifest.json
        ├── labs.csv
        └── patients.csv
        True csv 40

        The subject count defaulted to what this config's split fractions require, the format was
        chosen automatically, and the dry run confirms every event actually yields rows:

        >>> with tempfile.TemporaryDirectory() as d:
        ...     synthesize(cfg, d, seed=0, null_fraction=0.0).events
        shape: (3, 5)
        ┌──────────┬───────┬─────────┬──────────┬───────┐
        │ table    ┆ event ┆ rows_in ┆ rows_out ┆ error │
        │ ---      ┆ ---   ┆ ---     ┆ ---      ┆ ---   │
        │ str      ┆ str   ┆ i64     ┆ i64      ┆ str   │
        ╞══════════╪═══════╪═════════╪══════════╪═══════╡
        │ patients ┆ dob   ┆ 40      ┆ 40       ┆ null  │
        │ patients ┆ sex   ┆ 40      ┆ 40       ┆ null  │
        │ labs     ┆ lab   ┆ 160     ┆ 160      ┆ null  │
        └──────────┴───────┴─────────┴──────────┴───────┘

        With the default ``null_fraction`` some rows are dropped instead, because a null component
        of an interpolated code null-propagates under 0.7 and takes the row with it. Configs that
        write ``f"SEX//{$sex ?? 'UNK'}"`` keep those rows; the dry run shows which behaviour a
        given config has.
    """
    cfg = spec if isinstance(spec, MessyConfig) else MessyConfig.load(spec)
    if n_subjects is None:
        n_subjects = recommended_n_subjects(cfg)

    options = GenerationOptions(
        seed=seed,
        n_subjects=n_subjects,
        rows_per_subject=rows_per_subject,
        vocab_size=vocab_size,
        null_fraction=null_fraction,
        numeric_ranges=dict(numeric_ranges or {}),
    )
    dataset = generate(cfg, options)
    resolved = resolve_format(dataset.plan, fmt, dataset.frames)
    output_dir = Path(output_dir)
    files = write_dataset(dataset, output_dir, resolved)

    findings: list[Finding] = [
        Finding(
            "warning",
            prefix,
            f"{dropped} key combination(s) dropped at the metadata row cap; codes built from them "
            f"will have no description. Lower vocab_size to fit.",
        )
        for prefix, dropped in sorted(dataset.truncated_metadata.items())
    ]
    events = None
    if validate:
        findings.extend(check_plan(cfg, dataset.plan, resolved))
        events = dry_run(cfg, output_dir)
        findings.extend(findings_from_dry_run(events))

    return SynthesisResult(
        dataset=dataset,
        output_dir=output_dir,
        files=files,
        fmt=resolved,
        findings=findings,
        events=events,
    )

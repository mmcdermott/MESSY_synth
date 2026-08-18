"""Run a whole MEDS-Extract ETL over synthetic data and assert the result is a real MEDS dataset.

This is the payoff. :func:`smoke_test` generates synthetic sources for a MESSY config, invokes
``meds-extract-run`` over them, and then checks the output is actually a populated MEDS dataset —
not merely that the command exited 0.

That distinction is the whole point. MEDS-Extract exits 0 in several states that are not success:
a lenient time cast can null every row of a table and drop it with a warning; a dangling join key
silently deletes rows; a re-run into an existing output directory skips every stage and returns 0
without executing anything. A smoke test that only checks the return code passes vacuously in all
three cases, so :func:`check_output` inspects the artifacts instead.

Use it as a CI regression test for an ETL config::

    from MESSY_synth.smoke import smoke_test

    def test_etl_still_works(tmp_path):
        result = smoke_test("path/to/messy.yaml", tmp_path)
        assert result.ok, result.summary()
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from .synth import synthesize
from .validate import Finding

logger = logging.getLogger(__name__)

#: Default wall-clock limit for one ETL invocation.
DEFAULT_TIMEOUT_S = 1800


@dataclass(frozen=True)
class SmokeResult:
    """The outcome of one end-to-end run.

    Attributes:
        spec: The spec that was run.
        input_dir: Where the synthetic sources were written.
        output_dir: Where the MEDS dataset was written.
        returncode: The ETL process's exit status, or None if it was never started.
        findings: Problems found, most severe first.
        n_events: Total rows across ``data/*/*.parquet``.
        n_subjects: Distinct subjects in ``metadata/subject_splits.parquet``.
        n_codes: Rows in ``metadata/codes.parquet``.
        n_described_codes: Codes carrying a non-null description.
        log_tail: The last lines of the ETL's output, for diagnosis.
    """

    spec: str
    input_dir: Path
    output_dir: Path
    returncode: int | None = None
    findings: list[Finding] = field(default_factory=list)
    n_events: int = 0
    n_subjects: int = 0
    n_codes: int = 0
    n_described_codes: int = 0
    log_tail: str = ""

    @property
    def ok(self) -> bool:
        """Whether the run produced a valid, populated MEDS dataset.

        Returns:
            True if nothing at error level was found.
        """
        return not any(f.level == "error" for f in self.findings)

    def summary(self) -> str:
        """Render a human-readable one-paragraph summary.

        Returns:
            The rendered summary.
        """
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"[{status}] {self.spec}",
            f"    exit={self.returncode} events={self.n_events} subjects={self.n_subjects} "
            f"codes={self.n_codes} (described: {self.n_described_codes})",
        ]
        lines.extend(f"    {f}" for f in self.findings)
        if not self.ok and self.log_tail:
            lines.append("    --- ETL log tail ---")
            lines.extend(f"    {line}" for line in self.log_tail.splitlines()[-15:])
        return "\n".join(lines)


def run_etl(
    spec: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess:
    """Invoke ``meds-extract-run`` over pre-staged raw input.

    Two environment details are load-bearing and easy to get wrong:

    - The runner shells out to ``MEDS_transform-stage``, so the interpreter's ``bin/`` directory
      must be on ``PATH`` or every stage dies with return code 127 and an opaque error.
    - MEDS-Transforms marks completed stages under ``<output_dir>/.logs``. Re-running into a
      populated directory logs "All stages are already complete" and returns 0 without doing
      anything, so ``output_dir`` must not already exist.

    Args:
        spec: A registered pipeline name, ``pkg://`` reference, or path to a MESSY file.
        input_dir: Directory holding the raw (here, synthetic) source files.
        output_dir: Where to write the MEDS dataset. Must not already exist.
        timeout: Wall-clock limit in seconds.

    Returns:
        The completed process, with stdout and stderr captured.

    Raises:
        FileExistsError: If ``output_dir`` already exists, which would make the run a no-op.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} already exists and is non-empty. MEDS-Transforms would skip every "
            f"stage and exit 0 without running, so the smoke test would pass vacuously."
        )

    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])

    cmd = [
        "meds-extract-run",
        f"spec={spec}",
        f"output_dir={output_dir}",
        "do_download=false",
        f"input_dir={input_dir}",
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
        check=False,
    )


def check_output(output_dir: str | Path) -> tuple[list[Finding], dict[str, int]]:
    """Inspect a MEDS output directory and report whether it is genuinely populated.

    Args:
        output_dir: The ETL's output directory.

    Returns:
        The findings and a stats mapping with ``n_events``, ``n_subjects``, ``n_codes``, and
        ``n_described_codes``.

    Examples:
        >>> with tempfile.TemporaryDirectory() as d:
        ...     findings, stats = check_output(d)
        ...     print(findings[0])
        ...     stats["n_events"]
        ERROR    output: no data/ directory: the ETL produced no MEDS cohort at all.
        0
    """
    output_dir = Path(output_dir)
    findings: list[Finding] = []
    stats = {"n_events": 0, "n_subjects": 0, "n_codes": 0, "n_described_codes": 0}

    data_dir = output_dir / "data"
    if not data_dir.is_dir():
        findings.append(
            Finding("error", "output", "no data/ directory: the ETL produced no MEDS cohort at all.")
        )
        return findings, stats

    shards = sorted(data_dir.glob("*/*.parquet"))
    if not shards:
        findings.append(Finding("error", "data/", "no shard parquet files were written."))
    else:
        stats["n_events"] = sum(pl.read_parquet(p).height for p in shards)
        if stats["n_events"] == 0:
            findings.append(
                Finding("error", "data/", f"{len(shards)} shard(s) written but every one is empty.")
            )

    splits_fp = output_dir / "metadata" / "subject_splits.parquet"
    if not splits_fp.is_file():
        findings.append(Finding("error", "metadata/", "subject_splits.parquet is missing."))
    else:
        splits = pl.read_parquet(splits_fp)
        stats["n_subjects"] = splits["subject_id"].n_unique()
        empty = [
            row["split"] for row in splits.group_by("split").len().iter_rows(named=True) if row["len"] == 0
        ]
        if empty:
            findings.append(Finding("error", "metadata/", f"empty split(s): {empty}."))

    codes_fp = output_dir / "metadata" / "codes.parquet"
    if not codes_fp.is_file():
        findings.append(Finding("error", "metadata/", "codes.parquet is missing."))
    else:
        codes = pl.read_parquet(codes_fp)
        stats["n_codes"] = codes.height
        if "description" in codes.columns:
            stats["n_described_codes"] = int(codes["description"].is_not_null().sum())
        if codes.height == 0:
            findings.append(Finding("error", "metadata/", "codes.parquet is empty."))

    if not (output_dir / "metadata" / "dataset.json").is_file():
        findings.append(Finding("warning", "metadata/", "dataset.json is missing."))

    return findings, stats


def smoke_test(
    spec: str | Path,
    workdir: str | Path,
    *,
    n_subjects: int | None = None,
    rows_per_subject: int = 4,
    seed: int = 0,
    fmt: str = "auto",
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
) -> SmokeResult:
    """Generate synthetic sources for a MESSY config, run its ETL, and check the result.

    Args:
        spec: A registered pipeline name, ``pkg://`` reference, or path to a MESSY file.
        workdir: Scratch directory. ``<workdir>/raw`` and ``<workdir>/meds`` are created under it.
        n_subjects: Subject-universe size; defaults to what the config's splits require.
        rows_per_subject: Rows per subject in each timed event table.
        seed: Master seed.
        fmt: Source file format; see :data:`~MESSY_synth.writer.FORMATS`.
        numeric_ranges: Per-column numeric range overrides.
        timeout: Wall-clock limit for the ETL, in seconds.

    Returns:
        The :class:`SmokeResult`.

    Raises:
        TypeError: If ``spec`` is an in-memory config rather than something resolvable by path.

    Examples:
        >>> from MEDS_extract.config import MessyConfig
        >>> smoke_test(MessyConfig.parse({"t": {"e": {"code": "X", "time": None}}}), "/tmp")
        Traceback (most recent call last):
            ...
        TypeError: smoke_test needs a spec `meds-extract-run` can resolve...
    """
    if not isinstance(spec, str | Path):
        raise TypeError(
            f"smoke_test needs a spec `meds-extract-run` can resolve — a registered name, a pkg:// "
            f"reference, or a path to a MESSY file — not a {type(spec).__name__}. The ETL runs in a "
            f"subprocess, so an in-memory config cannot be handed to it; write it to a file first."
        )

    workdir = Path(workdir)
    input_dir = workdir / "raw"
    output_dir = workdir / "meds"

    synth = synthesize(
        spec,
        input_dir,
        n_subjects=n_subjects,
        rows_per_subject=rows_per_subject,
        seed=seed,
        fmt=fmt,
        numeric_ranges=numeric_ranges,
    )
    findings = list(synth.findings)

    try:
        proc = run_etl(spec, input_dir, output_dir, timeout=timeout)
    except subprocess.TimeoutExpired:
        findings.append(Finding("error", "etl", f"timed out after {timeout}s."))
        return SmokeResult(str(spec), input_dir, output_dir, None, findings)
    except FileExistsError as e:
        findings.append(Finding("error", "etl", str(e)))
        return SmokeResult(str(spec), input_dir, output_dir, None, findings)

    log_tail = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        findings.append(Finding("error", "etl", f"meds-extract-run exited {proc.returncode}."))

    out_findings, stats = check_output(output_dir)
    findings.extend(out_findings)

    order = {"error": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: (order.get(f.level, 3), f.where))

    return SmokeResult(
        spec=str(spec),
        input_dir=input_dir,
        output_dir=output_dir,
        returncode=proc.returncode,
        findings=findings,
        log_tail=log_tail,
        **stats,
    )

"""Command-line interface: ``MESSY-synth <spec> -o <dir>``.

The CLI is a thin wrapper over :func:`MESSY_synth.synth.synthesize`, with one addition worth
knowing about: it exits non-zero when validation finds an error. That makes it usable directly as a
CI smoke test for an ETL config — regenerate, and fail the build if any event stopped producing
rows.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .synth import synthesize
from .writer import FORMATS

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.

    Examples:
        >>> parser = build_parser()
        >>> args = parser.parse_args(["messy.yaml", "-o", "out"])
        >>> args.spec, args.output_dir, args.format, args.seed
        ('messy.yaml', 'out', 'auto', 0)

        Numeric range overrides are parsed into a mapping:

        >>> parser.parse_args(["m.yaml", "-o", "o", "-R", "anchor_year=1950:2000"]).numeric_range
        [('anchor_year', (1950.0, 2000.0))]
    """
    parser = argparse.ArgumentParser(
        prog="MESSY-synth",
        description=(
            "Generate synthetic raw source data matching the structure a MEDS-Extract v0.7 "
            "MESSY config expects. All generated values are fake; only the shape is real."
        ),
    )
    parser.add_argument(
        "spec",
        help="Registered pipeline name, pkg:// reference, or path to a MESSY yaml file.",
    )
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to write source files into.")
    parser.add_argument(
        "-n",
        "--n-subjects",
        type=int,
        default=None,
        help="Subject-universe size. Defaults to the smallest count that safely fills every split.",
    )
    parser.add_argument(
        "-r", "--rows-per-subject", type=int, default=4, help="Rows per subject in timed event tables."
    )
    parser.add_argument("-v", "--vocab-size", type=int, default=8, help="Values per categorical vocabulary.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Master seed; output is deterministic.")
    parser.add_argument(
        "--null-fraction",
        type=float,
        default=0.05,
        help="Fraction of nullable values emitted as nulls, to exercise null-handling paths.",
    )
    parser.add_argument(
        "-f", "--format", choices=FORMATS, default="auto", help="Output format for the source files."
    )
    parser.add_argument(
        "-R",
        "--numeric-range",
        action="append",
        type=_parse_range,
        default=[],
        metavar="COL=LO:HI",
        help="Override the numeric range for a column, e.g. anchor_year=1950:2000. Repeatable.",
    )
    parser.add_argument("--explain", action="store_true", help="Print the inferred schema per column.")
    parser.add_argument("--no-validate", action="store_true", help="Skip static checks and the dry run.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print errors.")
    return parser


def _parse_range(text: str) -> tuple[str, tuple[float, float]]:
    """Parse a ``COL=LO:HI`` numeric-range override.

    Args:
        text: The raw argument text.

    Returns:
        The column name and its bounds.

    Raises:
        argparse.ArgumentTypeError: If the text is malformed.

    Examples:
        >>> _parse_range("anchor_year=1950:2000")
        ('anchor_year', (1950.0, 2000.0))
        >>> _parse_range("bogus")
        Traceback (most recent call last):
            ...
        argparse.ArgumentTypeError: Expected COL=LO:HI, got 'bogus'
    """
    column, _, bounds = text.partition("=")
    lo, _, hi = bounds.partition(":")
    if not column or not lo or not hi:
        raise argparse.ArgumentTypeError(f"Expected COL=LO:HI, got {text!r}")
    try:
        return column, (float(lo), float(hi))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Expected numeric bounds in {text!r}: {e}") from e


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        0 on success, 1 if validation found an error.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
    )

    result = synthesize(
        args.spec,
        args.output_dir,
        n_subjects=args.n_subjects,
        rows_per_subject=args.rows_per_subject,
        vocab_size=args.vocab_size,
        seed=args.seed,
        null_fraction=args.null_fraction,
        fmt=args.format,
        numeric_ranges=dict(args.numeric_range),
        validate=not args.no_validate,
    )

    if not args.quiet:
        print(f"\nWrote {len(result.files)} file(s) to {result.output_dir} as {result.fmt}:")
        print(result.dataset.summary())
        if args.explain:
            print("\nInferred source schema:")
            print(explain(result))
        if result.events is not None:
            print("\nPer-event dry run:")
            print(result.events)

    for finding in result.findings:
        print(finding, file=sys.stderr)

    if not result.ok:
        print(
            "\nValidation failed. The generated data will not produce a complete MEDS dataset.",
            file=sys.stderr,
        )
        return 1
    return 0


def explain(result) -> str:
    """Render the inferred per-column schema as a readable table.

    Args:
        result: A :class:`~MESSY_synth.synth.SynthesisResult`.

    Returns:
        The rendered text.
    """
    lines: list[str] = []
    for table in result.dataset.plan.tables:
        lines.append(f"\n  {table.prefix}  ({table.n_rows} rows, {table.kind})")
        for column in table.columns:
            c = column.constraint
            extras = []
            if c.datetime_formats:
                extras.append("fmt=" + ",".join(c.datetime_formats))
            if c.temporal_role:
                extras.append(f"role={c.temporal_role}")
            if column.pool_id:
                extras.append("shared-pool" + (" (covering)" if column.covers_pool else ""))
            if not c.effective_nullable:
                extras.append("not-null")
            lines.append(f"      {column.name:<32} {c.kind.name:<13} {' '.join(extras)}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - entry point
    raise SystemExit(main())

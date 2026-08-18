"""Run the MESSY_synth smoke test across a set of MEDS-Extract v0.7 ETL configs.

This is the demonstration harness: point it at any number of MESSY files and it generates synthetic
sources for each, runs the real ETL over them, and prints a pass/fail matrix. No credentials and no
real data are involved — every input is invented from the config itself.

Usage::

    uv run python demo/smoke_matrix.py NAME=path/to/messy.yaml [NAME=...] [--workdir DIR]

Each argument is a ``label=path`` pair so the output table stays readable.
"""

from __future__ import annotations

import argparse
import logging
import traceback
from pathlib import Path

from MESSY_synth.smoke import smoke_test


def main(argv: list[str] | None = None) -> int:
    """Run the matrix.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        0 if every config passed, else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+", metavar="LABEL=PATH", help="Configs to test.")
    parser.add_argument("--workdir", required=True, help="Scratch directory for generated data.")
    parser.add_argument("-n", "--n-subjects", type=int, default=None)
    parser.add_argument("-r", "--rows-per-subject", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    workdir = Path(args.workdir)
    rows: list[tuple[str, str, str]] = []
    failed = False

    for item in args.specs:
        label, _, path = item.partition("=")
        label = label or path
        print(f"\n{'=' * 78}\n=== {label}\n{'=' * 78}", flush=True)
        try:
            result = smoke_test(
                path,
                workdir / label,
                n_subjects=args.n_subjects,
                rows_per_subject=args.rows_per_subject,
                timeout=args.timeout,
            )
            print(result.summary(), flush=True)
            status = "PASS" if result.ok else "FAIL"
            detail = (
                f"{result.n_events} events, {result.n_subjects} subjects, "
                f"{result.n_codes} codes ({result.n_described_codes} described)"
            )
            failed |= not result.ok
        except Exception as e:
            traceback.print_exc()
            status, detail = "ERROR", f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
            failed = True
        rows.append((label, status, detail))

    width = max(len(r[0]) for r in rows)
    print(f"\n\n{'=' * 78}\n=== SUMMARY\n{'=' * 78}")
    for label, status, detail in rows:
        print(f"  {label:<{width}}  {status:<5}  {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

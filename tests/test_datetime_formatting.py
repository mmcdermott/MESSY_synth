"""Columns the config parses with ``strptime`` are rendered with its formats, quickly."""

import time

from MEDS_extract.config import MessyConfig

from MESSY_synth import GenerationOptions, generate


def config(*times):
    """Build a config with one timed event per ``time`` expression."""
    events = {f"e{i}": {"code": f"X{i}", "time": time} for i, time in enumerate(times)}
    return MessyConfig.parse(
        {
            "etl": {"dataset_name": "Formatting", "raw_dataset_version": "1"},
            "t": {"_defaults": {"subject_id": "$pid"}, **events},
        }
    )


def test_a_column_rotates_through_its_formats_by_row():
    """Row ``i`` is rendered with the ``i``-th format, cycling, including Chrono's ``%.f``."""
    formats = ("%Y-%m-%d %H:%M:%S%.f", "%d/%m/%Y %H:%M")
    cfg = config(*(f'$ts::?"{fmt}"' for fmt in formats))
    frame = generate(cfg, GenerationOptions(n_subjects=50, seed=0, null_fraction=0.0)).frames["t"]
    for k, fmt in enumerate(formats):
        values = frame["ts"].gather_every(len(formats), offset=k)
        assert values.len() > 0
        assert values.str.to_datetime(fmt, strict=False).null_count() == 0, fmt


def test_formatting_a_column_does_not_dominate_generation():
    """Rendering a ``strptime`` column costs little next to generating it as a raw datetime.

    Formatting one value at a time made generation about 10x slower; the ratio is loose so that it only
    catches that kind of regression, not timing noise.
    """
    options = GenerationOptions(n_subjects=5_000, seed=0)

    def best_of_3(cfg):
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            generate(cfg, options)
            times.append(time.perf_counter() - t0)
        return min(times)

    raw = best_of_3(config("$ts"))
    formatted = best_of_3(config('$ts::"%Y-%m-%d %H:%M:%S"'))
    assert formatted < 3 * raw, f"formatted {formatted:.2f}s vs raw {raw:.2f}s"

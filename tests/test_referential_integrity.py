"""Cross-file invariants that make generated data actually extractable.

Each of these failure modes is silent in MEDS-Extract: a dangling join key nulls ``subject_id`` and
the row is dropped without an error, and a ``_metadata`` dictionary that misses the event's
vocabulary just produces null descriptions. They are therefore asserted here directly rather than
being left to the end-to-end test, which would only report the symptom.
"""

from __future__ import annotations

import pytest
from MEDS_extract.config import MessyConfig
from MEDS_extract.io import resolve_source_files

from MESSY_synth import GenerationOptions, generate, synthesize
from MESSY_synth.constraints import ValueKind

#: A config exercising every structural feature the generator has to satisfy at once: a join to a
#: dimension table, an aggregated join, derived `_table.cols`, nested prefixes, a metadata
#: dictionary with a composite (partially matched) code, static events, and birth/death.
FEATURE_CONFIG = {
    "etl": {
        "dataset_name": "Features",
        "raw_dataset_version": "1",
        "split_fracs": {"train": 0.5, "tuning": 0.25, "held_out": 0.25},
    },
    "_defaults": {"subject_id": "$pid"},
    "hosp/patients": {
        "_table": {
            "join": {"hosp/admissions": {"key": "pid", "cols": {"deathtime": "min"}}},
            "cols": {"final_dod": "$deathtime ?? $dod"},
        },
        "sex": {"code": 'f"SEX//{$sex}"', "time": None},
        "dob": {"code": "MEDS_BIRTH", "time": '$dob::"%Y-%m-%d"'},
        "death": {"code": "MEDS_DEATH", "time": '$final_dod::?"%Y-%m-%d %H:%M:%S"'},
    },
    "hosp/admissions": {
        "admit": {"code": 'f"ADMIT//{$admission_type}"', "time": '$admittime::"%Y-%m-%d %H:%M:%S"'},
    },
    "hosp/labs": {
        "_table": {"join": {"hosp/stays": {"key": "stay_id", "cols": ["pid"]}}},
        "lab": {
            "code": 'f"LAB//{$itemid}//{$uom}"',
            "time": '$charttime::"%Y-%m-%d %H:%M:%S"',
            "numeric_value": "$valuenum",
            "_metadata": {"hosp/d_labitems": {"itemid": "$itemid", "description": "$label"}},
        },
    },
}


@pytest.fixture
def features_cfg() -> MessyConfig:
    """Return the parsed feature-coverage config.

    Returns:
        The parsed config.
    """
    return MessyConfig.parse(FEATURE_CONFIG)


def test_every_table_the_config_needs_is_generated(features_cfg):
    """Event tables, join targets, and metadata dictionaries all get a file."""
    dataset = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    expected = set(features_cfg.needed_source_columns()) | set(features_cfg.events_by_metadata_prefix())
    assert set(dataset.frames) == expected


def test_columns_match_what_meds_extract_will_read(features_cfg):
    """Each generated frame carries exactly the columns MEDS-Extract plans to project."""
    dataset = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    for prefix, columns in features_cfg.needed_source_columns().items():
        assert set(dataset.frames[prefix].columns) == set(columns), prefix


def test_all_subject_columns_share_one_universe(features_cfg):
    """Every table's subject ids are drawn from the same pool, so events merge onto one subject."""
    dataset = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    universe = set(range(1, 17))
    seen = 0
    for table in dataset.plan.tables:
        if table.subject_column is None:
            continue
        seen += 1
        values = set(dataset.frames[table.prefix][table.subject_column].drop_nulls().to_list())
        assert values <= universe, table.prefix
    assert seen >= 3


def test_join_keys_resolve_on_the_far_side(features_cfg):
    """No join key dangles: MEDS-Extract left joins would silently drop those rows."""
    dataset = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    for table in features_cfg.event_tables:
        if table.join is None:
            continue
        left = dataset.frames[table.input_prefix]
        right = dataset.frames[table.join.input_prefix]
        for lkey, rkey in zip(table.join.left_on, table.join.right_on, strict=True):
            missing = set(left[lkey].drop_nulls().to_list()) - set(right[rkey].drop_nulls().to_list())
            assert not missing, f"{table.input_prefix}.{lkey} -> {table.join.input_prefix}.{rkey}"


def test_metadata_dictionary_covers_the_event_vocabulary(features_cfg):
    """The dictionary holds every code component the event table can emit."""
    dataset = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    events = set(dataset.frames["hosp/labs"]["itemid"].drop_nulls().to_list())
    dictionary = set(dataset.frames["hosp/d_labitems"]["itemid"].to_list())
    assert events <= dictionary
    assert events


def test_birth_precedes_events_and_death_follows_them(features_cfg):
    """Each subject's timeline is ordered, which downstream MEDS tooling assumes."""
    dataset = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    for timeline in dataset.timelines.values():
        assert timeline.birth < timeline.window_start < timeline.window_end
        assert timeline.death is None or timeline.death > timeline.window_end


def test_generation_is_deterministic(features_cfg):
    """The same config and seed reproduce the same data."""
    a = generate(features_cfg, GenerationOptions(seed=7, n_subjects=16))
    b = generate(features_cfg, GenerationOptions(seed=7, n_subjects=16))
    for prefix, frame in a.frames.items():
        assert frame.equals(b.frames[prefix]), prefix


def test_a_different_seed_changes_the_data(features_cfg):
    """Seeding actually varies the output, so a smoke test can be re-rolled."""
    a = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    b = generate(features_cfg, GenerationOptions(seed=2, n_subjects=16))
    assert not a.frames["hosp/labs"].equals(b.frames["hosp/labs"])


def test_written_files_resolve_through_meds_extracts_own_resolver(features_cfg, tmp_path):
    """MEDS-Extract can find every table, in exactly one layout, at the paths we wrote."""
    result = synthesize(features_cfg, tmp_path, n_subjects=16, seed=1)
    assert result.ok, [str(f) for f in result.findings]
    for prefix in features_cfg.needed_source_columns():
        found = resolve_source_files(tmp_path, prefix)
        assert len(found) == 1, f"{prefix} resolved to {found}"


def test_every_event_yields_rows(features_cfg, tmp_path):
    """The dry run confirms no event silently produces nothing."""
    result = synthesize(features_cfg, tmp_path, n_subjects=16, seed=1, null_fraction=0.0)
    assert result.events is not None
    empty = result.events.filter(result.events["rows_out"] == 0)
    assert empty.height == 0, empty


def test_subject_ids_are_int64_castable(features_cfg):
    """MEDS-Extract casts subject ids with strict=True, so they must be true integers."""
    dataset = generate(features_cfg, GenerationOptions(seed=1, n_subjects=16))
    for table in dataset.plan.tables:
        if table.subject_column is None:
            continue
        column = next(c for c in table.columns if c.name == table.subject_column)
        assert column.constraint.kind is ValueKind.SUBJECT_ID
        assert dataset.frames[table.prefix][table.subject_column].dtype.is_integer()


#: A config whose `kind` column is compared against more literals than `vocab_size` allows, so the
#: materialized pool ends up longer than the size the plan asked for.
OVERSIZED_POOL_CONFIG = {
    "etl": {"dataset_name": "Oversized", "raw_dataset_version": "1"},
    "labs": {
        "_defaults": {"subject_id": "$pid"},
        "lab": {
            "code": 'f"LAB//{$kind}"',
            "time": '$ts::"%Y-%m-%d"',
            "_metadata": {"d_kinds": {"kind": "$kind", "description": "$label"}},
        },
        "other": {
            "code": " else ".join(f'"C{i}" if $kind == "K{i}"' for i in range(6)) + ' else "OTHER"',
            "time": '$ts::"%Y-%m-%d"',
        },
    },
}


def test_pool_coverage_survives_a_pool_longer_than_its_planned_size():
    """A dictionary covers the whole vocabulary even when the pool outgrows the requested size.

    Pools are sized during planning, but the materialized pool can be longer: compared-against
    literals, coalesce fallbacks and regex-derived members are all added on top of the requested
    ``vocab_size``. Sizing the dictionary from the planned number would truncate it, and the codes
    built from the missing values would come back with null descriptions and no error anywhere.
    """
    cfg = MessyConfig.parse(OVERSIZED_POOL_CONFIG)
    dataset = generate(cfg, GenerationOptions(seed=0, n_subjects=16, vocab_size=3))
    events = set(dataset.frames["labs"]["kind"].drop_nulls().to_list())
    dictionary = set(dataset.frames["d_kinds"]["kind"].to_list())
    assert len(events) > 3, "the pool should have outgrown vocab_size for this test to mean anything"
    assert events <= dictionary, events - dictionary


#: A config deriving subject ids by hashing a text column, rather than reading an integer one.
HASHED_SUBJECT_CONFIG = {
    "etl": {"dataset_name": "Hashed", "raw_dataset_version": "1"},
    "_defaults": {"subject_id": "hash($MRN)"},
    "patients": {"sex": {"code": 'f"SEX//{$sex}"', "time": None}},
    "labs": {"lab": {"code": 'f"LAB//{$itemid}"', "time": '$ts::"%Y-%m-%d"'}},
}


def test_hashed_subject_keys_are_shared_and_sized_by_subject_count():
    """A column hashed into the subject id gets one value per requested subject, shared by table.

    ``hash($MRN)`` makes ``MRN`` stand in one-to-one for a subject. Treating it as an ordinary
    categorical would size it by ``vocab_size`` — so the dataset would quietly contain
    ``vocab_size`` subjects instead of the requested number — and would give each table its own
    vocabulary, describing two disjoint sets of people.
    """
    cfg = MessyConfig.parse(HASHED_SUBJECT_CONFIG)
    dataset = generate(cfg, GenerationOptions(seed=0, n_subjects=25))
    patients = set(dataset.frames["patients"]["MRN"].drop_nulls().to_list())
    labs = set(dataset.frames["labs"]["MRN"].drop_nulls().to_list())
    assert len(patients) == 25
    assert patients == labs


def test_a_config_meds_extract_rejects_is_reported_not_raised(tmp_path):
    """A config that cannot be loaded produces a finding, not a traceback.

    Surfacing broken ETL configs is the point of a smoke test across a fleet of them, so an
    unloadable config has to read as one failing result. MEDS-Extract validates tables lazily, via a
    ``cached_property``, so the rejection does not happen until the tables are materialized — which
    is why it is easy to let it escape from somewhere deep in generation instead.
    """
    import yaml

    from MESSY_synth.smoke import smoke_test

    # A self-join: every joined column already exists on the left side, which 0.7 rejects outright.
    cfg_fp = tmp_path / "messy.yaml"
    cfg_fp.write_text(
        yaml.safe_dump(
            {
                "etl": {"dataset_name": "Broken", "raw_dataset_version": "1"},
                "ops": {
                    "_defaults": {"subject_id": "$pid"},
                    "_table": {"join": {"ops": {"key": "pid", "cols": {"age": "min"}}}},
                    "e": {"code": "X", "time": None},
                },
            }
        )
    )
    result = smoke_test(cfg_fp, tmp_path / "run")
    assert not result.ok
    assert result.returncode is None, "the ETL should never have been started"
    assert any("could not be loaded" in f.message and "ops" in f.message for f in result.findings), (
        result.findings
    )


#: A plain (non-aggregated) join to a table that also carries its own events — the shape that makes
#: a repeated key on the target side fan the join out.
FANOUT_CONFIG = {
    "etl": {"dataset_name": "Fanout", "raw_dataset_version": "1"},
    "_defaults": {"subject_id": "$pid"},
    "admissions": {
        "admit": {"code": "ADMIT", "time": '$admittime::"%Y-%m-%d"'},
    },
    "diagnoses": {
        "_table": {"join": {"admissions": {"key": "hadm_id", "cols": ["dischtime"]}}},
        "dx": {"code": 'f"DX//{$icd}"', "time": '$dischtime::"%Y-%m-%d"'},
    },
}


def test_a_plain_join_does_not_fan_out(tmp_path):
    """A join to an event-bearing target preserves the referencing table's row count.

    MEDS-Extract uses a plain left join. If the target's key repeats — which it will whenever the target has
    many rows and the key is drawn from a small pool — one referencing row becomes many, each carrying a
    different arbitrary right-hand row. The event count inflates and rows get attributed to whichever
    admission happened to sort first, with no error anywhere.
    """
    cfg = MessyConfig.parse(FANOUT_CONFIG)
    result = synthesize(cfg, tmp_path, n_subjects=16, rows_per_subject=4, seed=0)
    diagnoses = result.dataset.frames["diagnoses"]
    joined = next(t for t in cfg.event_tables if t.input_prefix == "diagnoses").scan(tmp_path).collect()
    assert joined.height == diagnoses.height, f"{diagnoses.height} rows fanned out to {joined.height}"
    assert joined["dischtime"].null_count() == 0, "every join key should resolve on the far side"


#: One column times both an ordinary event and MEDS_DEATH — the shape that lets a death role
#: contaminate a column that most subjects must still have a value for.
SHARED_TIME_CONFIG = {
    "etl": {"dataset_name": "SharedTime", "raw_dataset_version": "1"},
    "adm": {
        "_defaults": {"subject_id": "$pid"},
        "_table": {
            "cols": {
                "_disch": '$dischtime::"%Y-%m-%d %H:%M:%S"',
                "_died": '$status != "alive"',
                "dod": "$_disch if $_died",
            }
        },
        "admission": {"code": "HOSPITAL_ADMISSION", "time": '$admittime::"%Y-%m-%d %H:%M:%S"'},
        "discharge": {"code": "HOSPITAL_DISCHARGE", "time": "$_disch"},
        "death": {"code": "MEDS_DEATH", "time": "$dod"},
    },
}


def test_a_death_role_does_not_empty_a_column_shared_with_an_ordinary_event(tmp_path):
    """A column timing both a normal event and MEDS_DEATH is generated as a normal timestamp.

    Birth and death timestamps are placed on the subject's timeline rather than sampled, and a
    death date is null for every subject who survives. When the config routes an ordinary event's
    timestamp through the same raw column — HiRID does, via ``date_of_death: '$datetime if $_died'``
    — honouring the death role would null that column for most subjects and silently delete most of
    the ordinary event's rows, while every check still reported success.
    """
    cfg = MessyConfig.parse(SHARED_TIME_CONFIG)
    result = synthesize(cfg, tmp_path, n_subjects=20, rows_per_subject=1, seed=0, null_fraction=0.0)
    assert result.dataset.frames["adm"]["dischtime"].null_count() == 0
    yields = {row["event"]: row["rows_out"] for row in result.events.iter_rows(named=True)}
    assert yields["discharge"] == yields["admission"] == 20, yields

"""Unit tests for the dashboard's data layer (dashboard/data.py) and the
pure helpers in its render module (dashboard/app.py).

WHY THIS FILE EXISTS. `dashboard/` shipped with no tests at all and was
excluded from the coverage gate (`--cov=cooling_twin` measures `src/`
only), so a single wrong predicate survived into a live panel:
`equifinality_candidates` kept every candidate carrying a `parameters`
key instead of only the BEHAVIOURAL ones, and the what-if panel reported
a structural interval built from 41 parameter sets when 3 are
behavioural -- 38 of them fits L6.8 had explicitly REJECTED. Measured on
`Fox_education_Claude` at +1.0 K, the displayed band was [-4.43%,
-0.14%] against a true [-3.15%, -3.10%]: 77x too wide, under help text
calling it "the only one that measures structural uncertainty".

The reports were never affected (counterfactual_2016.json records
n_parameter_sets = 3) because `scripts/run_counterfactual.py` filtered
correctly. That is the whole lesson: the bug was a DIVERGED COPY of a
correct loop, so the fix was to delete the copy, and the test that
matters most below is `test_dashboard_and_script_agree_exactly` --
a duplicate cannot drift when there is only one of it.

Artifact-dependent tests are skipped rather than failed when
`reports/calibration_runs/` has not been generated: a fresh clone has no
artifacts and 05_ENGINEERING_STANDARDS.md is explicit that H must never
be blocked. Nothing here loads BDG2 or solves an ODE, so the file runs
in well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard import data

EQUIFINALITY_BUILDING = "Fox_education_Claude"

# L6.8 wrote 41 restarts for this building, of which exactly 3 clear the
# 5% behavioural threshold. Hard-coded rather than recomputed: the point
# is to pin the number the panel displays against the artifact on disk,
# and a test that derives both sides from the same code proves nothing.
EXPECTED_BEHAVIOURAL_SETS = 3
EXPECTED_TOTAL_CANDIDATES = 41

REQUIRED_PARAMETER_NAMES = frozenset(
    {
        "ua_envelope_w_per_m2k",
        "r_internal_ratio",
        "internal_gain_w_per_m2",
        "vent_flow_kg_per_s",
        "t_setpoint_c",
    }
)


def _artifact_paths() -> list[Path]:
    """Equifinality artifacts on disk, newest-sorted like the loaders."""
    return sorted(data.ARTIFACTS_DIR.glob("equifinality_*.json"), reverse=True)


requires_equifinality = pytest.mark.skipif(
    not _artifact_paths(),
    reason="no equifinality_*.json on disk -- run `make equifinality` first",
)
requires_hybrid = pytest.mark.skipif(
    not (data.ARTIFACTS_DIR / "hybrid_2016.json").exists(),
    reason="no hybrid_2016.json on disk -- run `make hybrid` first",
)
requires_gate = pytest.mark.skipif(
    not (data.ARTIFACTS_DIR / "gate_2017_opened.json").exists(),
    reason="no gate_2017_opened.json on disk -- run `make reproduce` first",
)


# --------------------------------------------------------------------- #
# D1 -- the equifinality ensemble. The regression this file was written for.
# --------------------------------------------------------------------- #


@requires_equifinality
def test_only_behavioural_sets_are_returned() -> None:
    """Rejected parameter sets must never reach the ensemble.

    This is the assertion that fails on the pre-fix code (it returned
    41). A rejected set is one the study measured and ruled out; feeding
    it to the structural interval reports a disagreement between fits
    that L6.8 already decided are not equally good.
    """
    candidates = data.equifinality_candidates(EQUIFINALITY_BUILDING)
    assert len(candidates) == EXPECTED_BEHAVIOURAL_SETS


@requires_equifinality
def test_returned_sets_are_flagged_behavioural_in_the_artifact() -> None:
    """Every returned set traces to a `behavioural: true` candidate.

    Stronger than the count above: a loader that returned the wrong 3
    of 41 would pass the count test and fail this one.
    """
    candidates = data.equifinality_candidates(EQUIFINALITY_BUILDING)
    behavioural_on_disk = []
    for path in _artifact_paths():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("building_id") != EQUIFINALITY_BUILDING:
            continue
        behavioural_on_disk = [
            candidate["parameters"]
            for candidate in record["candidates"]
            if candidate.get("behavioural")
        ]
        break
    assert candidates == behavioural_on_disk


@requires_equifinality
def test_dashboard_and_script_agree_exactly() -> None:
    """The panel and the report must be built from the SAME parameter sets.

    The bug this file documents was a diverged copy of
    `run_counterfactual.behavioural_parameter_sets`. Now that the copy
    is gone this is trivially true -- which is the point: it fails the
    moment anyone reintroduces a private loader in the dashboard.
    """
    from run_counterfactual import behavioural_parameter_sets

    script_sets, source = behavioural_parameter_sets(
        data.ARTIFACTS_DIR, EQUIFINALITY_BUILDING
    )
    assert data.equifinality_candidates(EQUIFINALITY_BUILDING) == script_sets
    assert source is not None


@requires_equifinality
def test_the_artifact_really_does_hold_rejected_sets() -> None:
    """Guards the guard.

    If the artifact ever contained only behavioural candidates, every
    test above would pass against a broken loader too. Assert the
    hazard is still present before trusting the tests that avoid it.
    """
    record = json.loads(_artifact_paths()[0].read_text(encoding="utf-8"))
    candidates = record["candidates"]
    assert len(candidates) == EXPECTED_TOTAL_CANDIDATES
    rejected = [c for c in candidates if not c.get("behavioural")]
    assert len(rejected) == EXPECTED_TOTAL_CANDIDATES - EXPECTED_BEHAVIOURAL_SETS
    assert all("parameters" in c for c in rejected), (
        "rejected candidates must still carry parameters -- if they did not, "
        "the original wrong predicate would have been harmless and this "
        "regression suite would be testing nothing"
    )


@requires_equifinality
def test_returned_sets_are_usable_by_the_twin() -> None:
    """Each set is a name-keyed dict `CalibratedTwin` accepts.

    `EquifinalityStudy.from_dict` exposes `BehaviouralSet.parameters` as
    a POSITIONAL TUPLE ordered by `parameter_names`, which `CalibratedTwin`
    rejects with "calibrated parameters missing [...]". Anyone refactoring
    this loader towards that class needs to convert; this test says so
    before the panel does.
    """
    for candidate in data.equifinality_candidates(EQUIFINALITY_BUILDING):
        assert isinstance(candidate, dict)
        assert set(candidate) >= REQUIRED_PARAMETER_NAMES


def test_building_without_a_study_gets_an_empty_list() -> None:
    """UNMEASURED is reported as empty, never as a zero-width interval.

    The panel renders "n/a" for this case. A study-less building
    silently receiving a zero spread would claim the twin's structural
    uncertainty had been measured and found negligible.
    """
    assert data.equifinality_candidates("Bull_education_Luke") == []
    assert data.equifinality_candidates("Hog_education_Cathleen") == []


# --------------------------------------------------------------------- #
# D3 -- artifact lookups that a render function must not silently drop.
# --------------------------------------------------------------------- #


@requires_hybrid
def test_cathleen_is_genuinely_absent_from_the_hybrid_artifact() -> None:
    """Pins the condition the "no bar for ..." notice exists to report.

    If a later hybrid run adds Cathleen this test fails, which is the
    correct outcome: the notice would then be dead code and the caption
    describing it would be stale.
    """
    hybrid = data.load_artifact("hybrid_2016.json")
    assert data.artifact_building_record(hybrid, "Hog_education_Cathleen") is None
    assert data.artifact_building_record(hybrid, "Fox_education_Claude") is not None


def test_artifact_building_record_returns_none_for_unknown_building() -> None:
    """A miss is `None`, not a KeyError and not a fabricated record."""
    artifact = {"buildings": [{"building_id": "A", "value": 1}]}
    assert data.artifact_building_record(artifact, "A") == {"building_id": "A", "value": 1}
    assert data.artifact_building_record(artifact, "B") is None
    assert data.artifact_building_record({}, "A") is None


@requires_gate
def test_roster_only_lists_buildings_with_a_gate_record() -> None:
    """`Fox_education_Theodore` is in buildings.yaml but must not be selectable.

    It was screened out BEFORE calibration (ADR-012) and has no frozen
    parameters, so a live twin for it would have to be fabricated.
    """
    roster = data.building_roster()
    ids = {entry["building_id"] for entry in roster}
    assert "Fox_education_Theodore" not in ids
    assert ids == {
        "Fox_education_Claude",
        "Bull_education_Luke",
        "Hog_education_Cathleen",
    }
    for entry in roster:
        assert entry["role"] in data.ROLE_LABELS


@requires_gate
def test_every_roster_building_has_the_keys_the_gate_cards_read() -> None:
    """`render_gate_cards` indexes these directly; a miss is a page crash."""
    gate = data.load_artifact("gate_2017_opened.json")
    for entry in data.building_roster():
        record = data.artifact_building_record(gate, entry["building_id"])
        assert record is not None
        assert "relative_improvement_pct" in record
        for split in ("train", "test"):
            assert {"cvrmse_pct", "nmbe_pct"} <= set(record[split])
        assert isinstance(record["test"]["passed"], bool)


def test_missing_artifact_raises_rather_than_returning_a_default() -> None:
    """The page must say the artifact is absent, not invent a number."""
    with pytest.raises(FileNotFoundError, match="not found"):
        data.load_artifact("no_such_artifact_2016.json")

"""Persistence and control boundaries of the observation-only learner."""

import json

from custom_components.better_thermostat.utils.calibration.mpc_v2 import (
    MpcV2Input,
    MpcV2Params,
    MpcV2State,
    compute_mpc_v2,
    export_mpc_v2_state,
    import_mpc_v2_state,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
    ValveResponseLearner,
)
from tests.unit.mpc_v2.test_report_response import sparse_room


def test_candidates_never_change_mpc_commands_or_cap(monkeypatch):
    learned = ValveResponseLearner()
    sparse_room(learned)
    assert learned.diagnostics()["accepted_episodes"] > 0
    now = 100000 + 36 * 3600
    monkeypatch.setattr("time.time", lambda: now)
    baseline, observing = MpcV2State(), MpcV2State(response=learned)
    for i in range(100):
        inp = MpcV2Input(
            key="room",
            target_temp_C=21.5,
            current_temp_C=20.5 + i / 100,
            outdoor_temp_C=10,
            max_opening_pct=35,
            response_managed=True,
        )
        a, baseline = compute_mpc_v2(inp, MpcV2Params(), baseline, now=now + i * 60)
        b, observing = compute_mpc_v2(
            inp, MpcV2Params(learn_valve_response=True), observing, now=now + i * 60
        )
        assert a.valve_percent == b.valve_percent <= 35
        assert b.diagnostics.response_curve["control_active"] is False
    restored = import_mpc_v2_state(
        json.loads(json.dumps(export_mpc_v2_state(observing)))
    )
    assert (
        restored.response.diagnostics()["accepted_episodes"]
        == learned.diagnostics()["accepted_episodes"]
    )
    assert restored.response.reports == []


def test_legacy_evidence_is_preserved_without_becoming_new_confidence():
    learner = ValveResponseLearner()
    raw = {
        "v": 1,
        "source": "room|weather|valve",
        "samples": {"35": [[1, 2, 0.05, 0.002]]},
    }
    learner.restore(raw, 1000)
    assert learner.export()["legacy"] == raw
    assert learner.diagnostics()["accepted_episodes"] == 0
    assert learner.control_curve(0.05) is None


def test_expired_duplicate_and_malformed_episodes_are_not_counted():
    learner = ValveResponseLearner()
    sparse_room(learner)
    raw = learner.export()
    raw["episodes"] = [raw["episodes"][0], raw["episodes"][0], {"end": "bad"}]
    restored = ValveResponseLearner()
    restored.restore(raw, 100000 + 36 * 3600)
    assert restored.diagnostics()["accepted_episodes"] == 1
    expired = ValveResponseLearner()
    expired.restore(raw, 100000 + 20 * 86400)
    assert expired.diagnostics()["accepted_episodes"] == 0
    for bad in (None, {}, {"v": 2, "episodes": None, "gaps": None}):
        ValveResponseLearner().restore(bad, 1000)


def test_changed_source_discards_incompatible_evidence():
    learner = ValveResponseLearner()
    sparse_room(learner)
    learner.bind_source("different|outdoor|valve", 100000 + 36 * 3600)
    assert learner.diagnostics()["accepted_episodes"] == 0
    assert learner.reports == []

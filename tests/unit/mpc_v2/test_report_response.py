"""Sparse reports must retain input history without fabricating measurements."""

import json
import math
import random

import pytest

from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
    ResponseObservation,
    ValveResponseLearner,
)


def feed(learner, minute, temperature, *, report=None, valve=20, valid=True):
    now = 100000.0 + minute * 60
    learner.observe(
        ResponseObservation(
            now, now if report is None else report, temperature, 10, valve, valid
        )
    )


def test_quiet_sensor_retains_commands_without_inventing_reports():
    learner = ValveResponseLearner()
    feed(learner, 0, 21)
    for minute in range(1, 61):
        feed(learner, minute, 21, report=100000, valve=minute % 35)
    d = learner.diagnostics()
    assert d["pending_reports"] == 1
    assert d["pending_commands"] > 30
    assert d["status"] == "waiting_for_report"
    feed(learner, 61, 21.3)
    assert learner.diagnostics()["pending_reports"] == 2
    assert learner.control_curve(0.05) is None


def test_explicit_dropout_breaks_episode_and_cannot_bridge_recovery():
    learner = ValveResponseLearner()
    feed(learner, 0, 21)
    feed(learner, 30, 21.4)
    feed(learner, 31, 21.4, valid=False)
    assert learner.diagnostics()["pending_reports"] == 0
    feed(learner, 32, 21.4, report=101800)
    assert learner.diagnostics()["pending_reports"] == 0
    feed(learner, 80, 21.5)
    assert learner.diagnostics()["pending_reports"] == 1


def test_missed_heartbeat_and_out_of_order_reports_do_not_add_evidence():
    learner = ValveResponseLearner()
    feed(learner, 0, 21)
    feed(learner, 91, 21, report=100000)
    assert learner.diagnostics()["status"] == "missing_heartbeat"
    assert learner.diagnostics()["pending_reports"] == 0
    feed(learner, 120, 21.2)
    feed(learner, 121, 21.3, report=100000)
    assert learner.diagnostics()["pending_reports"] == 1


def sparse_room(
    learner,
    hours=36,
    *,
    constant=False,
    solar=False,
    drop=False,
    cap=100,
    jitter=False,
    noise=0,
    lag=2,
    boiler=False,
):
    """Different plant, sensor lag, quantization and event-triggered reports."""
    room = sensor = 21.0
    heat = 0.0
    last_reading = 21.0
    last_report = 100000.0
    reports = 0
    rng = random.Random(71)
    heartbeat = 3600
    for minute in range(hours * 60 + 1):
        # Changing commands never hold for 30 minutes during heating.
        phase = minute % 240
        valve = (
            20
            if constant
            else (0 if phase < 90 else [10, 35, 70, 100, 20, 50][(phase // 10) % 6])
        )
        valve = min(valve, cap)
        power = 0.04 * (1 + 0.3 * math.sin(minute / 100)) if boiler else 0.04
        heat += (power * min(valve / 35, 1) - heat) * (1 - math.exp(-1 / 12))
        outdoor = 10 + 1.5 * math.sin(minute / 900)
        room += (
            heat
            - 0.0017 * (room - outdoor)
            + (0.03 * math.sin(minute / 40) if solar else 0)
        )
        sensor += (room - sensor) * (1 - math.exp(-1 / lag))
        reading = round(sensor + rng.gauss(0, noise), 2)
        now = 100000.0 + minute * 60
        if abs(reading - last_reading) >= 0.5 or now - last_report >= heartbeat:
            if not drop or minute % 240 < 120:
                last_report, last_reading = now, reading
                reports += 1
                heartbeat = rng.randint(55, 65) * 60 if jitter else 3600
        learner.observe(
            ResponseObservation(now, last_report, last_reading, outdoor, valve)
        )
    return reports


def test_learns_candidate_from_threshold_reports_and_varying_commands():
    learner = ValveResponseLearner()
    reports = sparse_room(learner)
    d = learner.diagnostics()
    assert reports < 200  # Most controller ticks have no new measurement.
    assert d["accepted_episodes"] >= 2
    assert d["heat_K_min"][7] == pytest.approx(0.04, abs=0.018)
    assert d["heat_K_min"][-1] == pytest.approx(0.04, abs=0.018)
    assert d["control_active"] is False
    assert learner.control_curve(0.05) is None


def test_constant_opening_cannot_identify_a_curve():
    learner = ValveResponseLearner()
    sparse_room(learner, constant=True)
    assert learner.diagnostics()["accepted_episodes"] == 0
    assert learner.diagnostics()["saturation_pct"] is None


def test_persistence_keeps_completed_evidence_but_not_pending_interval():
    learner = ValveResponseLearner()
    sparse_room(learner)
    original = learner.diagnostics()["accepted_episodes"]
    assert original > 0
    restored = ValveResponseLearner()
    restored.restore(json.loads(json.dumps(learner.export())), 100000 + 36 * 3600)
    assert restored.diagnostics()["accepted_episodes"] == original
    assert restored.diagnostics()["pending_reports"] == 0
    assert restored.control_curve(0.05) is None


def test_handling_jump_clears_interval_and_quarantines_recovery():
    learner = ValveResponseLearner()
    feed(learner, 0, 21)
    feed(learner, 1, 24)
    assert learner.diagnostics()["status"] == "temperature_jump"
    feed(learner, 2, 23.5)
    assert learner.diagnostics()["pending_reports"] == 0


def test_noisy_jittered_reports_keep_unvisited_range_unknown():
    learner = ValveResponseLearner()
    sparse_room(learner, hours=48, cap=35, jitter=True, noise=0.015, lag=4)
    d = learner.diagnostics()
    assert d["accepted_episodes"] >= 1
    assert d["heat_K_min"][7] == pytest.approx(0.04, abs=0.02)
    assert all(v is None for v in d["heat_K_min"][8:])
    assert d["saturation_pct"] is None


@pytest.mark.parametrize("options", [{"solar": True}, {"drop": True}])
def test_confounded_or_missing_reports_do_not_create_a_curve(options):
    learner = ValveResponseLearner()
    sparse_room(learner, **options)
    assert learner.diagnostics()["accepted_episodes"] == 0


def test_boiler_variation_never_activates_a_candidate():
    learner = ValveResponseLearner()
    sparse_room(learner, hours=48, boiler=True, lag=8, jitter=True)
    assert learner.control_curve(0.05) is None
    assert learner.diagnostics()["control_active"] is False

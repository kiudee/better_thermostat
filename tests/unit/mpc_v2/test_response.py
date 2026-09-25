"""Behavioral tests with a separate saturating radiator and delayed sensor."""

import json
import math
import random

import pytest

from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
    ResponseObservation,
    ValveResponseLearner,
)


class Room:
    def __init__(self, *, lag=3, noise=0.015):
        self.t = 100000.0
        self.room = 21.0
        self.sensor = self.room
        self.heat = 0.0
        self.lag = lag
        self.noise = noise
        self.rng = random.Random(42)

    def hold(self, learner, valve, minutes=100, *, drop=False):
        for _ in range(minutes):
            # Independent simulated physics: full heat at 35%, ten-minute lag.
            self.heat += (0.06 * min(valve / 35, 1) - self.heat) * (
                1 - math.exp(-1 / 10)
            )
            self.room += self.heat - 0.002 * (self.room - 10)
            self.sensor += (self.room - self.sensor) * (1 - math.exp(-1 / self.lag))
            self.t += 60
            reported = self.t - 3600 if drop else self.t
            learner.observe(
                ResponseObservation(
                    self.t,
                    reported,
                    self.sensor + self.rng.gauss(0, self.noise),
                    10,
                    valve,
                )
            )


def learned_curve(openings=(15, 35, 50, 65, 80, 100), repeats=6):
    learner = ValveResponseLearner()
    room = Room()
    for _ in range(repeats):
        for opening in openings:
            room.hold(learner, 0)
            room.hold(learner, opening)
    return learner, room


def test_learns_saturating_response_with_sensor_lag_and_noise():
    learner, _ = learned_curve()
    diag = learner.diagnostics()
    for pct in (15, 35, 50, 65, 80, 100):
        i = pct // 5
        assert diag["independent_holds"][i] == 6
        assert diag["heat_K_min"][i] == pytest.approx(
            0.06 * min(pct / 35, 1), abs=0.009
        )
        assert diag["lower_K_min"][i] < diag["upper_K_min"][i]
    assert diag["saturation_pct"] == 35


def test_one_long_hold_does_not_create_false_confidence():
    learner = ValveResponseLearner()
    room = Room()
    room.hold(learner, 0)
    room.hold(learner, 35, minutes=100)
    diag = learner.diagnostics()
    assert diag["independent_holds"][7] == 1
    assert diag["lower_K_min"][7] is None
    assert diag["saturation_pct"] is None
    assert learner.control_curve(0.05) is None


def test_unvisited_high_openings_cannot_establish_true_cap():
    learner, _ = learned_curve((15, 35), repeats=6)
    diag = learner.diagnostics()
    assert diag["heat_K_min"][-1] is None
    assert diag["upper_K_min"][-1] is None
    assert diag["saturation_pct"] is None
    curve, _ = learner.control_curve(0.05)
    assert all(b >= a for a, b in zip(curve, curve[1:]))
    assert curve[-1] > curve[7]


def test_dropout_and_airing_do_not_train_or_bridge_recovery():
    learner = ValveResponseLearner()
    room = Room()
    room.hold(learner, 0)
    room.hold(learner, 35, drop=True)
    assert sum(learner.diagnostics()["independent_holds"]) == 0
    room.hold(learner, 35)
    assert sum(learner.diagnostics()["independent_holds"]) == 0  # lost off baseline
    learner.interrupt(room.t, "window_open")
    room.hold(learner, 35)
    assert sum(learner.diagnostics()["independent_holds"]) == 0


def test_outlier_does_not_become_a_heating_sample():
    learner = ValveResponseLearner()
    room = Room()
    room.hold(learner, 0)
    room.hold(learner, 35, minutes=40)
    learner.observe(ResponseObservation(room.t + 60, room.t + 60, 40, 10, 35))
    assert learner.status == "temperature_jump"
    assert not learner.samples


def test_persistence_drops_pending_hold_and_invalid_or_expired_evidence():
    learner, room = learned_curve((35,), repeats=3)
    raw = json.loads(json.dumps(learner.export()))
    restored = ValveResponseLearner()
    restored.restore(raw, room.t)
    assert restored.diagnostics()["independent_holds"][7] == 3
    assert restored.baseline is None
    assert restored.control_curve(0.05) is not None
    expired = ValveResponseLearner()
    expired.restore(raw, room.t + 15 * 86400)
    assert expired.control_curve(0.05) is None
    restored.restore(
        {"v": 1, "samples": {"35": [[0, room.t, float("nan"), 0.002]]}}, room.t
    )
    assert restored.control_curve(0.05) is None


def test_learned_curve_reduces_overshoot_with_full_configured_cap():
    from custom_components.better_thermostat.utils.calibration.mpc_v2 import (
        MpcV2Input,
        MpcV2Params,
        MpcV2State,
        compute_mpc_v2,
    )

    learned, room = learned_curve()

    def run(adaptive):
        state = MpcV2State()
        if adaptive:
            state.response.restore(learned.export(), room.t)
        temp = measured = 20.0
        heat = 0.0
        valve = 0
        temperatures = []
        commands = []
        for minute in range(240):
            now = room.t + minute * 60 + 60
            output, state = compute_mpc_v2(
                MpcV2Input(
                    key="test",
                    target_temp_C=21.5,
                    current_temp_C=measured,
                    outdoor_temp_C=10,
                    max_opening_pct=100,
                    room_reported_at=now,
                    outdoor_reported_at=now,
                    applied_valve_pct=valve,
                ),
                MpcV2Params(learn_valve_response=adaptive),
                state,
                now=now,
            )
            valve = output.valve_percent
            commands.append(valve)
            heat += (0.06 * min(valve / 35, 1) - heat) * (1 - math.exp(-1 / 10))
            temp += heat - 0.002 * (temp - 10)
            measured += (temp - measured) * (1 - math.exp(-1 / 3))
            temperatures.append(temp)
        return max(temperatures) - 21.5, temperatures[-1], commands

    baseline, _, _ = run(False)
    adaptive, final, commands = run(True)
    assert adaptive < baseline * 0.65, (baseline, adaptive)
    assert adaptive < 0.35
    assert final == pytest.approx(21.5, abs=0.25)
    assert max(commands) <= 100


def test_confidence_stays_unknown_when_samples_are_duplicated_on_import():
    learner = ValveResponseLearner()
    sample = [1000, 2000, 0.06, 0.002]
    learner.restore({"v": 1, "samples": {"35": [sample] * 20}}, 2100)
    assert learner.diagnostics()["independent_holds"][7] == 1
    assert learner.diagnostics()["lower_K_min"][7] is None


def test_adaptive_controller_preserves_cap_and_evidence_across_restart(monkeypatch):
    from custom_components.better_thermostat.utils.calibration.mpc_v2 import (
        MpcV2Input,
        MpcV2Params,
        MpcV2State,
        compute_mpc_v2,
        export_mpc_v2_state,
        import_mpc_v2_state,
    )

    learned, room = learned_curve()
    monkeypatch.setattr("time.time", lambda: room.t + 1)
    state = MpcV2State()
    state.response.restore(learned.export(), room.t)
    inp = MpcV2Input(
        key="test",
        target_temp_C=24,
        current_temp_C=18,
        outdoor_temp_C=5,
        max_opening_pct=35,
    )
    params = MpcV2Params(learn_valve_response=True)
    first, state = compute_mpc_v2(inp, params, state, now=room.t + 1)
    assert first.diagnostics.response_curve["control_active"]
    assert first.valve_percent <= 35
    exported = json.loads(json.dumps(export_mpc_v2_state(state)))
    restored = import_mpc_v2_state(exported, params)
    second, restored = compute_mpc_v2(inp, params, restored, now=room.t + 61)
    assert second.valve_percent <= 35
    assert (
        second.diagnostics.response_curve["independent_holds"]
        == first.diagnostics.response_curve["independent_holds"]
    )
    assert second.diagnostics.T_rad_hat is None
    assert math.isfinite(second.diagnostics.heat_rate_hat_K_min)


@pytest.mark.parametrize("lag", [1, 8, 15])
def test_slow_sensor_does_not_turn_one_hold_into_precise_flow(lag):
    learner = ValveResponseLearner()
    room = Room(lag=lag)
    for _ in range(3):
        room.hold(learner, 0)
        room.hold(learner, 35)
    diag = learner.diagnostics()
    assert diag["heat_K_min"][7] == pytest.approx(0.06, abs=0.014)
    assert diag["lower_K_min"][7] is None
    assert diag["saturation_pct"] is None


def test_changing_sensor_pairing_does_not_reuse_a_different_rooms_curve():
    learner, room = learned_curve((35,), repeats=3)
    assert learner.control_curve(0.05) is not None
    learner.bind_source("another_sensor|outside|another_valve", room.t)
    assert learner.control_curve(0.05) is None
    assert sum(learner.diagnostics()["independent_holds"]) == 0


def test_conflicting_operating_conditions_do_not_establish_saturation():
    learner = ValveResponseLearner()
    samples = {}
    for opening, heat in [(35, 0.08), (50, 0.04), (65, 0.04), (80, 0.04), (100, 0.04)]:
        samples[str(opening)] = [
            [1000 + i * 100, 1050 + i * 100, heat, 0.002] for i in range(6)
        ]
    learner.restore({"v": 1, "samples": samples}, 2000)
    assert learner.control_curve(0.05) is None
    assert learner.diagnostics()["saturation_pct"] is None

"""Forecast correction remains stable with sparse reports and an independent room."""

from dataclasses import dataclass
import math

import pytest

from custom_components.better_thermostat.utils.calibration.mpc_v2 import (
    MpcV2Input,
    MpcV2Params,
    compute_mpc_v2,
    state as state_module,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.flow_prior import (
    ValveFlowPrior,
)


@dataclass
class Scenario:
    name: str
    tau: float = 1387.6
    loss: float = 0.001
    power: float = 0.055
    lag: float = 12
    onset: float = 19.5
    near: float = 40
    threshold: float = 0.5
    event: str = "none"


def simulate(module, scenario, keep_trace=False):
    compute, Input, Params, Prior, State = module
    params = Params(flow_prior=Prior())
    params.plant.tau_room_min = scenario.tau
    state = None
    room = 20.8
    heat = 0
    opening = 0
    raw = room
    reported = 100000.0
    trace = []
    closed_after_drop = []
    interrupted = False
    for elapsed in range(0, 16 * 3600 + 1, 5):
        now = 100000 + elapsed
        hour = elapsed / 3600
        window = scenario.event == "window" and 6 <= hour < 6.25
        extra = (
            (
                0.006
                if scenario.event == "free_heat"
                else -0.004
                if scenario.event == "cold_load"
                else 0
            )
            if 5 <= hour < 9
            else 0
        )
        power = scenario.power * (
            1.5 if scenario.event == "supply" and 5 <= hour < 9 else 1
        )
        if elapsed:
            flow = 1 - math.exp(
                -math.log(50)
                * (max(0, opening - scenario.onset) / (scenario.near - scenario.onset))
                ** 1.3
            )
            heat += (power * flow - heat) * (5 / 60) / scenario.lag
            room += (
                heat
                - scenario.loss * (room - 10)
                + 0.001
                + extra
                - (0.08 if window else 0)
            ) * (5 / 60)
        fresh = abs(room - raw) >= scenario.threshold or now - reported >= 3600
        if fresh:
            raw, reported = round(room, 2), now
        target = 21.5 if hour < 14 else 19
        if scenario.event == "restart" and elapsed == 8 * 3600 and state:
            state = State.import_mpc_v2_state(State.export_mpc_v2_state(state), params)
        if window != interrupted:
            if state and state.controller:
                state.controller.record_applied_u(now, 0)
            opening = 0
            interrupted = window
        if fresh and window and state and state.controller:
            state.controller.observe_report(now, raw, learn_disturbance=False)
        if elapsed % 300 == 0 or fresh:
            output, state = compute(
                Input(
                    key="independent-simulation",
                    current_temp_C=raw,
                    target_temp_C=target,
                    outdoor_temp_C=10,
                    room_reported_at=reported,
                    window_open=window,
                    max_opening_pct=50,
                    applied_valve_pct=opening,
                    applied_input_managed=True,
                ),
                params,
                state,
                now=now,
            )
            opening = 0 if output is None else output.valve_percent
            if state.controller:
                state.controller.record_applied_u(
                    now, state.controller.valve_to_input(opening)
                )
                assert all(
                    math.isfinite(v) for v in state.controller.export_snapshot().x_hat
                )
            assert 0 <= opening <= 50
            if hour >= 14:
                closed_after_drop.append(opening)
        if elapsed % 60 == 0:
            trace.append(
                {
                    "hour": hour,
                    "room": room,
                    "target": target,
                    "opening": opening,
                    "raw": raw,
                    "fast": state.controller.dob.D_hat_K_per_min
                    if state.controller
                    else 0,
                    "planning": state.controller.dob.planning_rate
                    if state.controller
                    else 0,
                }
            )
    errors = [
        r["room"] - 21.5
        for r in trace
        if 3 <= r["hour"] < 14
        and not (scenario.event == "window" and 6 <= r["hour"] < 7.25)
    ]
    result = {
        "scenario": scenario.__dict__,
        "mean_error_C": sum(errors) / len(errors),
        "rmse_C": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "peak_C": max(errors),
        "low_C": min(errors),
        "final_opening": opening,
        "closed_on_target_drop": closed_after_drop[0] == 0,
    }
    if keep_trace:
        result["trace"] = trace
    return result


@pytest.mark.parametrize(
    "event", ["none", "free_heat", "cold_load", "supply", "window", "restart"]
)
def test_sparse_reports_control_heat_without_persistent_overshoot(event):
    module = (compute_mpc_v2, MpcV2Input, MpcV2Params, ValveFlowPrior, state_module)
    result = simulate(module, Scenario(event, event=event))
    # Bounds cover a 0.5 K report threshold, warm-up excluded, and room/model mismatch.
    assert result["rmse_C"] < 0.2
    assert result["peak_C"] < 0.45
    assert result["low_C"] > -0.3
    assert result["closed_on_target_drop"]
    assert result["final_opening"] == 0

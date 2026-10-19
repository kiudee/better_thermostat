"""A valve characteristic must survive different heating power on later days."""

import math

from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
    ValveResponseLearner,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.response_pool import (
    fit_short_periods,
)


def heating_period(index, power, *, confounded=False):
    start = 100000 + index * 86400
    room, heat = 21.0, 0.0
    reports, inputs = [], []
    for tick in range(1801):
        minute = tick / 10
        opening = 0
        if 15 <= minute < 55:
            opening = (15, 25, 35, 60)[int((minute - 15) // 10)]
        elif 80 <= minute < 110:
            opening = (60, 25, 35)[int((minute - 80) // 10)]
        outdoor = 10.0
        if not inputs or inputs[-1][1] != opening:
            inputs.append([start + tick * 6, opening, outdoor])
        if tick:
            heat += (power * min(1, max(0, (opening - 10) / 25)) - heat) / 12 / 10
            disturbance = 0.025 * math.sin(minute / 17) if confounded else 0.001
            room += (heat - 0.0015 * (room - outdoor) + disturbance) / 10
        if tick % 100 == 0:
            reports.append([start + tick * 6, round(room, 2)])
    return {"reports": reports, "inputs": inputs}


def test_shared_valve_shape_predicts_a_later_period_with_different_heat_supply():
    periods = [
        heating_period(i, power)
        for i, power in enumerate([0.02, 0.09, 0.035, 0.10, 0.055])
    ]
    candidate, reason = fit_short_periods(periods)
    assert candidate is not None, reason
    assert candidate["validation_rmse"] < 0.15
    assert candidate["model"] == "variable_supply"
    shape = candidate["relative_heat"]
    # Lag and deadzone remain confounded at the lowest openings.
    assert shape[3] is None
    assert shape[7] == shape[12] == 1.0
    assert shape[-1] is None
    learner = ValveResponseLearner()
    learner.restore({"v": 4, "episodes": [candidate]}, candidate["end"] + 1)
    restored = ValveResponseLearner()
    restored.restore(learner.export(), candidate["end"] + 2)
    assert restored.episodes[0] == candidate
    assert restored.diagnostics()["supply_models"][0]["relative_heat"] == shape
    assert restored.control_curve(0.04) is None


def test_unobserved_changes_in_heat_supply_within_a_period_are_rejected():
    periods = [
        heating_period(i, power, confounded=True)
        for i, power in enumerate([0.02, 0.09, 0.035, 0.10, 0.055])
    ]
    candidate, _ = fit_short_periods(periods)
    assert candidate is None


def test_a_later_bad_period_cannot_change_the_shape_to_pass_validation():
    periods = [
        heating_period(i, power)
        for i, power in enumerate([0.02, 0.09, 0.035, 0.10, 0.055])
    ]
    for row in periods[-1]["reports"][10:]:
        row[1] += 1
    candidate, reason = fit_short_periods(periods)
    assert candidate is None
    assert reason == "supply_model_validation_mismatch"


def test_cooling_only_tail_does_not_validate_the_valve_shape():
    from custom_components.better_thermostat.utils.calibration.mpc_v2.response_supply import (
        fit_variable_supply,
    )

    periods = [
        heating_period(i, power)
        for i, power in enumerate([0.02, 0.09, 0.035, 0.10, 0.055])
    ]
    start = periods[-1]["reports"][0][0]
    for row in periods[-1]["inputs"]:
        if row[0] >= start + 55 * 60:
            row[1] = 0
    candidate, reason = fit_variable_supply(periods)
    assert candidate is None
    assert reason == "insufficient_supply_validation"

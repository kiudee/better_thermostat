"""Short clean periods must survive until independent validation is possible."""

import math

import pytest

from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
    ResponseObservation,
    ValveResponseLearner,
)


def short_period(learner, index, *, heated=True, confused=False):
    start = 100000.0 + index * 86400
    room = 21.0 + index * 0.05
    radiator = 0.0
    reported_at = start
    reading = room
    for minute in range(181):
        now = start + minute * 60
        opening = (
            [10, 35, 70][((minute - 45) // 5 + index) % 3]
            if heated and 45 <= minute < 70
            else 0
        )
        outdoor = 10 + 0.5 * math.sin(minute / 90 + index)
        if minute:
            radiator += (0.04 * min(opening / 35, 1) - radiator) * (
                1 - math.exp(-1 / 15)
            )
            disturbance = 0.001 * math.sin(index)
            if confused:
                disturbance += 0.025 * math.sin(minute / 17 + index)
            room += radiator - 0.0017 * (room - outdoor) + disturbance
        if minute % 15 == 0:
            reported_at, reading = now, round(room, 2)
        learner.observe(
            ResponseObservation(now, reported_at, reading, outdoor, opening)
        )
    learner.interrupt(start + 181 * 60, "window_open_or_unavailable")


def test_independent_short_heat_periods_produce_validated_observation():
    learner = ValveResponseLearner()
    for index in range(6):
        short_period(learner, index)
    d = learner.diagnostics()
    assert d["accepted_episodes"] >= 1
    assert d["heat_K_min"][10] == pytest.approx(0.04, abs=0.015)
    assert d["heat_K_min"][-1] is None
    assert not d["control_active"]


def test_short_heat_is_retained_without_premature_candidate():
    learner = ValveResponseLearner()
    short_period(learner, 0)
    d = learner.diagnostics()
    assert d["retained_periods"] == 1
    assert 24 <= d["retained_heating_min"] <= 26
    assert d["accepted_episodes"] == 0
    assert not learner.reports


def test_closed_periods_survive_restart_but_pending_reports_do_not():
    learner = ValveResponseLearner()
    short_period(learner, 0)
    learner.observe(ResponseObservation(190000, 190000, 21, 10, 0))
    restored = ValveResponseLearner()
    restored.restore(learner.export(), 190000)
    assert restored.diagnostics()["retained_periods"] == 1
    assert not restored.reports


@pytest.mark.parametrize("heated,confused", [(False, False), (True, True)])
def test_unheated_or_confounded_short_periods_cannot_force_a_curve(heated, confused):
    learner = ValveResponseLearner()
    for index in range(6):
        short_period(learner, index, heated=heated, confused=confused)
    assert learner.diagnostics()["accepted_episodes"] == 0


def test_handling_or_uncertain_delivery_drops_the_current_period():
    learner = ValveResponseLearner()
    for minute in range(0, 121, 15):
        now = 100000 + minute * 60
        learner.observe(ResponseObservation(now, now, 21, 10, 30))
    learner.interrupt(108000, "valve_command_uncertain")
    assert learner.diagnostics()["retained_periods"] == 0


def test_executor_result_cannot_move_evidence_to_a_different_source():
    learner = ValveResponseLearner()
    learner.bind_source("original", 100000)
    learner.defer_fitting = True
    for index in range(3):
        short_period(learner, index)
    request = learner.pending_fits[-1]
    from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
        fit_response_request,
    )

    result = fit_response_request(request)
    assert result[0] is not None
    learner.bind_source("replacement", 400000)
    learner.finish_request(request, result)
    assert not learner.episodes
    assert not learner.periods


def test_reusing_an_accepted_batch_cannot_increase_evidence_count():
    learner = ValveResponseLearner()
    learner.defer_fitting = True
    for index in range(3):
        short_period(learner, index)
    request = learner.pending_fits[-1]
    from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
        fit_response_request,
    )

    result = fit_response_request(request)
    learner.finish_request(request, result)
    learner.finish_request(request, result)
    assert len(learner.episodes) == 1


def test_temperature_jump_cannot_archive_contaminated_pending_reports():
    learner = ValveResponseLearner()
    for minute in range(0, 121, 15):
        now = 100000 + minute * 60
        learner.observe(ResponseObservation(now, now, 21, 10, 30))
    learner.observe(ResponseObservation(107201, 107201, 25, 10, 30))
    assert not learner.periods


def test_restoration_rejects_duplicate_and_malformed_periods():
    learner = ValveResponseLearner()
    short_period(learner, 0)
    payload = learner.export()
    period = payload["periods"][0]
    payload["periods"] = [period, period, {"reports": None, "inputs": []}]
    restored = ValveResponseLearner()
    restored.restore(payload, 190000)
    assert len(restored.periods) == 1


def test_outdoor_updates_do_not_split_sustained_opening_coverage():
    from custom_components.better_thermostat.utils.calibration.mpc_v2.response_model import (
        period_exposure,
    )

    reports = [[1000, 21], [1120, 21.1]]
    inputs = [[1000 + seconds, 35, 10 + seconds / 100] for seconds in range(0, 120, 10)]
    inputs.append([1300, 100, 11])
    off, heated, maximum = period_exposure(reports, inputs)
    assert off == 0
    assert heated == 2
    assert maximum == 35

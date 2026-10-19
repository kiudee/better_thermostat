"""Sparse sensor corrections must not become permanent forecast disturbances."""

import pytest

from custom_components.better_thermostat.utils.calibration.mpc_v2 import (
    MpcV2Input,
    MpcV2Params,
    compute_mpc_v2,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.controller import (
    MpcV2Controller,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.flow_prior import (
    ValveFlowPrior,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.state import (
    import_mpc_v2_state,
)


@pytest.mark.parametrize(
    ("reported", "estimated", "radiator", "correction", "outdoor"),
    [
        (21.65, 21.687, 22.468, -0.0051, 10.4),
        (21.66, 21.655, 22.016, -0.0065, 12.0),
        (21.71, 21.75, 22.335, -0.0131, 11.9),
        (21.83, 21.692, 22.014, -0.0067, 10.9),
    ],
)
def test_transient_cooling_does_not_reopen_an_already_warm_room(
    reported, estimated, radiator, correction, outdoor
):
    params = MpcV2Params(flow_prior=ValveFlowPrior())
    params.plant.tau_room_min = 1387.6
    state = import_mpc_v2_state(
        {
            "last_percent": 0,
            "last_compute_ts": 1000,
            "snapshot": {
                "v": 3,
                "x_hat": [estimated, radiator],
                "D_hat_K_per_min": correction,
                "planning_disturbance": 0.0,
                "last_u": 0.0,
                "applied_u": 0.0,
                "last_t_s": 1300,
                "next_mpc_t_s": 1299,
                "last_reported_at": 1300,
                "outdoor_C": outdoor,
                "target_C": 21.5,
                "rg_v_C": 21.5,
                "flow_prior_identity": params.flow_prior.identity,
            },
        },
        params,
    )
    output, _ = compute_mpc_v2(
        MpcV2Input(
            key="sparse-room",
            current_temp_C=reported,
            target_temp_C=21.5,
            outdoor_temp_C=outdoor,
            room_reported_at=1300,
            max_opening_pct=50,
            applied_valve_pct=0,
            applied_input_managed=True,
        ),
        params,
        state,
        now=1300,
    )
    assert output is not None
    assert output.valve_percent == 0


def report_controller():
    params = MpcV2Params()
    params.plant.tau_room_min = 1387.6
    c = MpcV2Controller(params)
    c.step(1000, 21.5, 21.5, 10, room_reported_at=1000, applied_input_managed=True)
    c.record_applied_u(1000, 0)
    return c


def test_planning_learns_from_sparse_genuine_reports_only():
    c = report_controller()
    for t in (2800, 4600, 6400, 8200):
        # A warm room with a closed radiator implies sustained unmodelled heat.
        c.observe_report(t, 21.5)
    estimate = c.export_snapshot().planning_disturbance
    assert estimate is not None and 0.002 < estimate < 0.05
    c.observe_report(8200, 25)
    c.record_outdoor(8300, 11)
    c.record_applied_u(8400, 0.2)
    c.step(8500, 21.5, 21.5, 11, room_reported_at=8200, applied_input_managed=True)
    assert c.export_snapshot().planning_disturbance == estimate


def test_planning_learns_sustained_cooling_across_sparse_reports():
    c = report_controller()
    for index in range(1, 5):
        c.observe_report(1000 + index * 1800, 21.5 - 0.5 * index)
    estimate = c.export_snapshot().planning_disturbance
    assert estimate is not None and -0.05 < estimate < -0.002


def test_short_report_jump_does_not_become_a_persistent_load():
    c = report_controller()
    c.observe_report(1001, 21.0)
    # The fast correction responds to a jump; its forecast influence needs time.
    assert c.dob.D_hat_K_per_min < -0.01
    assert c.dob.planning_rate == 0


def test_uncertain_delivery_and_interruption_do_not_train_the_forecast():
    c = report_controller()
    c.observe_report(2800, 21.5)
    before = c.export_snapshot().planning_disturbance
    assert before != 0
    c.mark_input_uncertain(3000)
    c.record_applied_u(3100, 0)
    c.observe_report(4600, 23)
    assert c.export_snapshot().planning_disturbance == before
    c.observe_report(6400, 19, learn_disturbance=False)
    c.observe_report(8200, 21.5)
    assert c.export_snapshot().planning_disturbance == before


def test_report_planning_survives_restart_without_learning_the_unobserved_gap():
    c = report_controller()
    c.observe_report(2800, 21.5)
    snap = c.export_snapshot()
    assert snap.planning_disturbance != 0
    restored = MpcV2Controller(c.params)
    restored.restore_snapshot(snap)
    assert restored.export_snapshot().planning_disturbance == snap.planning_disturbance
    restored.observe_report(10000, 25)
    assert restored.export_snapshot().planning_disturbance == snap.planning_disturbance


def test_persistent_cooling_can_still_raise_heat_demand():
    c = report_controller()
    # A genuinely sustained sink has independent evidence in the planning state.
    c.dob.planning_filtered = -0.01
    c.dob.D_hat_K_per_min = -0.01
    command, _ = c.step(
        1300, 21.5, 21.5, 10, room_reported_at=1000, applied_input_managed=True
    )
    assert command > 0.1

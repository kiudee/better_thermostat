"""A manual actuator prior changes control units without inventing evidence."""

from dataclasses import replace
import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("daqp")

from custom_components.better_thermostat.calibration import _compute_mpc_v2_balance
from custom_components.better_thermostat.config_flow import _build_advanced_fields
from custom_components.better_thermostat.utils.calibration.mpc_v2 import (
    MpcV2Input,
    MpcV2Params,
    compute_mpc_v2,
    export_mpc_v2_state,
    import_mpc_v2_state,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.controller import (
    MpcV2Controller,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.flow_prior import (
    ValveFlowPrior,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.io import (
    MpcV2Diagnostics,
)
from custom_components.better_thermostat.utils.const import CalibrationType
from tests.unit.mpc_v2.test_dispatch import _make_bt, _trv_info
from tests.unit.mpc_v2.test_state_manager_live import _make_manager


def inp(**kwargs):
    return replace(
        MpcV2Input(
            key="room",
            current_temp_C=21,
            target_temp_C=22,
            outdoor_temp_C=10,
            room_reported_at=1000,
            applied_valve_pct=0,
            applied_input_managed=True,
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "flow,opening",
    [(0, 0), (1e-12, 0), (0.01, 0), (0.1, 21), (0.5, 25), (0.98, 40), (1, 50)],
)
def test_real_compute_boundary_maps_demand_and_keeps_off_closed(flow, opening):
    params = MpcV2Params(flow_prior=ValveFlowPrior(), learn_valve_response=True)
    diag = MpcV2Diagnostics(21, 25, 0, 480, 1)
    with patch.object(MpcV2Controller, "step", return_value=(flow, diag)):
        output, state = compute_mpc_v2(inp(max_opening_pct=50), params, now=1000)
    assert output.valve_percent == opening
    assert state.controller.optimiser.params.u_max == pytest.approx(0.9985799886)
    assert state.controller.governor.params.u_max == pytest.approx(0.9985799886)
    assert output.diagnostics.flow_prior["source"] == "manual_assumption"
    assert output.diagnostics.response_curve["control_active"] is False
    assert state.response.episodes == []
    # Issuing a recommendation does not pretend it was delivered.
    assert state.controller.export_snapshot().applied_u == 0


def test_quantization_is_monotone_bounded_and_chooses_nearest_realizable_flow():
    prior = ValveFlowPrior()
    for cap in [0, 18, 19.5, 20, 25.9, 40, 50, 100]:
        previous = 0
        for desired in np.linspace(0, 1, 201):
            command = prior.command(float(desired), cap)
            assert previous <= command <= int(cap)
            assert command == 0 or command > prior.onset_pct
            best_error = min(abs(prior.flow(v) - desired) for v in range(int(cap) + 1))
            assert abs(prior.flow(command) - desired) == pytest.approx(best_error)
            previous = command


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        ValveFlowPrior().command(bad)
    with pytest.raises(ValueError):
        ValveFlowPrior().flow(bad)
    with pytest.raises(ValueError):
        ValveFlowPrior(onset_pct=bad)


def test_completed_writes_use_flow_but_learning_keeps_physical_commands():
    prior = ValveFlowPrior()
    manager = _make_manager()
    _, state = compute_mpc_v2(
        inp(max_opening_pct=30), MpcV2Params(flow_prior=prior), now=1000
    )
    state.actuator_id = "climate.radiator"
    manager.set_mpc_v2_live("room", state)
    learner = manager.get_response_learner("temperature|weather|climate.radiator")
    learner.record_command = Mock(wraps=learner.record_command)
    # An acknowledged closing bump can exceed the currently configured cap.
    manager.record_response_command("climate.radiator", 45, 1001)
    assert state.controller.export_snapshot().applied_u == pytest.approx(prior.flow(45))
    learner.record_command.assert_called_once_with(1001, 45)
    requested_before = state.controller.export_snapshot().last_u
    manager.record_response_command("climate.radiator", 0, 1002)
    assert state.controller.export_snapshot().applied_u == 0
    assert state.controller.export_snapshot().last_u == requested_before
    manager.invalidate_response_command("climate.radiator", 1003)
    assert state.controller.export_snapshot().input_uncertain is True


def test_matching_snapshot_round_trip_and_changed_input_units_start_fresh():
    params = MpcV2Params(flow_prior=ValveFlowPrior())
    _, state = compute_mpc_v2(inp(), params, now=1000)
    state.controller.record_applied_u(1001, 0.5)
    raw = json.loads(json.dumps(export_mpc_v2_state(state)))
    same = import_mpc_v2_state(raw, params)
    assert same.controller.export_snapshot().applied_u == 0.5
    assert (
        same.controller.export_snapshot().flow_prior_identity == "weibull:19.5:40:1.3"
    )
    different = import_mpc_v2_state(raw, MpcV2Params())
    assert different.controller._initialised is False
    assert different.controller.export_snapshot().applied_u == 0
    _, linear = compute_mpc_v2(inp(), MpcV2Params(), now=1000)
    old = export_mpc_v2_state(linear)
    old["snapshot"].pop("flow_prior_identity")
    old["snapshot"]["v"] = 2
    assert import_mpc_v2_state(old, MpcV2Params()).controller._initialised is True
    assert import_mpc_v2_state(old, params).controller._initialised is False


def test_switching_prior_parameters_rebuilds_even_for_small_changes():
    prior = ValveFlowPrior()
    _, state = compute_mpc_v2(inp(), MpcV2Params(flow_prior=prior), now=1000)
    before = state.controller
    _, state = compute_mpc_v2(
        inp(), MpcV2Params(flow_prior=replace(prior, onset_pct=19)), state, now=1001
    )
    assert state.controller is not before


def test_over_target_solver_closes_without_numerical_flow_creep():
    output, _ = compute_mpc_v2(
        inp(current_temp_C=25, target_temp_C=19),
        MpcV2Params(flow_prior=ValveFlowPrior()),
        now=1000,
    )
    assert output.valve_percent == 0


@pytest.mark.parametrize("kwargs", [{"window_open": True}, {"heating_allowed": False}])
def test_existing_protection_gate_stays_in_charge(kwargs):
    output, _ = compute_mpc_v2(
        inp(**kwargs), MpcV2Params(flow_prior=ValveFlowPrior()), now=1000
    )
    assert output is None


@pytest.mark.parametrize(
    "enabled,direct,count,active",
    [
        (False, True, 1, False),
        (True, True, 1, True),
        (True, False, 1, False),
        (True, True, 2, False),
    ],
)
def test_dispatch_limits_prior_to_opted_in_single_direct_radiator(
    enabled, direct, count, active
):
    trvs = {
        f"climate.r{i}": _trv_info(
            f"climate.r{i}", current_temp=20, supports_valve=direct
        )
        for i in range(count)
    }
    for trv in trvs.values():
        trv.advanced["mpc_v2_flow_prior"] = enabled
    output, _ = _compute_mpc_v2_balance(_make_bt(real_trvs=trvs), "climate.r0")
    assert (output.diagnostics.flow_prior is not None) is active


def test_option_defaults_off_and_preserves_explicit_choice():
    for enabled in [False, True]:
        fields = _build_advanced_fields(
            sources=[{"mpc_v2_flow_prior": True}] if enabled else [],
            default_calibration=CalibrationType.DIRECT_VALVE_BASED,
            homematic=False,
            has_auto=False,
            support_valve=True,
        )
        key = next(k for k in fields if k.schema == "mpc_v2_flow_prior")
        assert key.default() is enabled

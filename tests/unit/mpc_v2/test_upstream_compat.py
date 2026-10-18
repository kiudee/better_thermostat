"""Upstream model updates retain timestamped observations and heat-rate units."""

import numpy as np
import pytest

from custom_components.better_thermostat.utils.calibration.mpc_v2 import MpcV2Params
from custom_components.better_thermostat.utils.calibration.mpc_v2.controller import (
    MpcV2Controller,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.flow_prior import (
    ValveFlowPrior,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2_internals.plant import (
    PlantParams,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2_internals.response_plant import (
    HeatResponsePlant,
)


def test_timestamped_half_year_gap_is_constant_cost(monkeypatch):
    c = MpcV2Controller(MpcV2Params())
    c.step(1000, 21, 21, 5, room_reported_at=1000)
    c.record_applied_u(1000, 0.5)
    original = c.plant_fine.continuous_dynamics
    calls = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(c.plant_fine, "continuous_dynamics", counted)
    now = 1000 + 180 * 86400
    _, diag = c.step(now, 21, 21, 5, room_reported_at=now)
    assert len(calls) == 1
    assert np.isfinite(diag.T_room_hat)
    assert 21 < diag.T_rad_hat < c.params.plant.T_water_C


def test_clock_recovery_preserves_mapping_and_accepts_subsequent_delivery():
    c = MpcV2Controller(MpcV2Params(flow_prior=ValveFlowPrior()))
    c.step(20000, 21, 22, 10, room_reported_at=20000, applied_input_managed=True)
    c.step(1000, 21, 22, 10, room_reported_at=1000, applied_input_managed=True)
    c.record_applied_u(1001, c.valve_to_input(30))
    c.observe_report(1300, 21.2)
    snap = c.export_snapshot()
    assert snap.last_reported_at == 1300
    assert snap.applied_u == pytest.approx(ValveFlowPrior().flow(30))
    assert snap.flow_prior_identity == ValveFlowPrior().identity
    assert all(np.isfinite(snap.x_hat))


def test_heat_rate_propagation_keeps_its_state_units_across_longer_steps():
    plant = HeatResponsePlant(PlantParams(), 30, 0.06, 0.001)
    initial = np.array([21.0, 0.03])
    expected = initial.copy()
    for _ in range(10):
        expected = plant.discrete_step(expected, 0.4, 10)
    actual = plant.propagate(initial, 0.4, 10, 300)
    assert actual == pytest.approx(expected)
    assert 0 <= actual[1] <= plant.heat_max


def test_saved_report_driven_disturbance_keeps_its_existing_bound():
    params = MpcV2Params(flow_prior=ValveFlowPrior())
    c = MpcV2Controller(params)
    c.step(1000, 21, 21, 10, room_reported_at=1000)
    c.dob.D_hat_K_per_min = -0.08
    restored = MpcV2Controller(params)
    restored.restore_snapshot(c.export_snapshot())
    assert restored.dob.D_hat_K_per_min == -0.08

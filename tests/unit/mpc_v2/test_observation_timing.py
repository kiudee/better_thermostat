"""Physical estimates follow elapsed time and genuine reports, not polling."""

import pytest

pytest.importorskip("daqp")

from custom_components.better_thermostat.utils.calibration.mpc_v2 import (
    MpcV2Input,
    MpcV2Params,
    compute_mpc_v2,
)
from custom_components.better_thermostat.utils.calibration.mpc_v2.controller import (
    MpcV2Controller,
)


def run_polling(interval):
    state = None
    output = None
    for elapsed in range(0, 1801, interval):
        output, state = compute_mpc_v2(
            MpcV2Input(
                key="room",
                target_temp_C=21.5,
                current_temp_C=21.5,
                outdoor_temp_C=13,
                room_reported_at=1000,
                applied_valve_pct=30,
            ),
            MpcV2Params(),
            state,
            now=1000 + elapsed,
        )
    assert output is not None
    return output.diagnostics


def test_polling_does_not_turn_a_held_reading_into_new_measurements():
    fast, slow = run_polling(30), run_polling(300)
    assert fast.T_rad_hat == pytest.approx(slow.T_rad_hat, abs=0.01)
    assert fast.T_room_hat == pytest.approx(slow.T_room_hat, abs=0.01)
    assert fast.D_hat_K_per_min == pytest.approx(slow.D_hat_K_per_min, abs=1e-6)


def test_a_completed_write_changes_prediction_from_its_timestamp_only():
    def run(extra_poll):
        c = MpcV2Controller(MpcV2Params())
        c.step(1000, 21, 21, 10, room_reported_at=1000, applied_input_managed=True)
        c.record_applied_u(1060, 1)
        if extra_poll:
            c.step(1120, 21, 21, 10, room_reported_at=1000, applied_input_managed=True)
        c.record_applied_u(1180, 0)
        return c.step(
            1600, 21, 21, 10, room_reported_at=1000, applied_input_managed=True
        )[1]

    a, b = run(False), run(True)
    assert a.T_rad_hat == pytest.approx(b.T_rad_hat, abs=1e-7)
    assert a.T_room_hat == pytest.approx(b.T_room_hat, abs=1e-7)


def test_unchanged_heartbeat_is_evidence_but_duplicate_timestamp_is_not():
    c = MpcV2Controller(MpcV2Params())
    c.step(1000, 21, 21, 10, room_reported_at=1000, applied_input_managed=True)
    c.record_applied_u(1000, 1)
    before = c.step(
        1600, 21, 21, 10, room_reported_at=1000, applied_input_managed=True
    )[1]
    c.observe_report(1600, 21)
    after = c.export_snapshot()
    assert abs(after.x_hat[0] - 21) < abs(before.T_room_hat - 21)
    c.observe_report(1600, 23)
    assert c.export_snapshot() == after


def test_report_identity_survives_a_snapshot_round_trip():
    c = MpcV2Controller(MpcV2Params())
    c.step(1000, 21, 21, 10, room_reported_at=1000, applied_input_managed=True)
    restored = MpcV2Controller(MpcV2Params())
    restored.restore_snapshot(c.export_snapshot())
    before = restored.export_snapshot()
    restored.observe_report(1000, 25)
    assert restored.export_snapshot() == before


def test_lowering_target_replans_immediately_without_resetting_physical_state():
    c = MpcV2Controller(MpcV2Params())
    high, _ = c.step(1000, 20, 25, 10, room_reported_at=1000)
    before = c.export_snapshot()
    low, _ = c.step(1000, 20, 16, 10, room_reported_at=1000)
    assert low < high
    assert c.export_snapshot().x_hat == before.x_hat


def test_uncertain_delivery_is_not_learned_as_an_external_heat_gain():
    c = MpcV2Controller(MpcV2Params())
    c.step(1000, 21, 21, 10, room_reported_at=1000, applied_input_managed=True)
    c.mark_input_uncertain(1060)
    c.record_applied_u(1300, 0)
    before = c.export_snapshot().D_hat_K_per_min
    c.observe_report(1600, 22)
    assert c.export_snapshot().D_hat_K_per_min == before


def test_event_loop_ordering_does_not_drop_a_genuine_report():
    c = MpcV2Controller(MpcV2Params())
    c.step(1000, 21, 21, 10, room_reported_at=1000, applied_input_managed=True)
    c.record_applied_u(1600.05, 0)
    before = c.export_snapshot().x_hat[0]
    c.observe_report(1600, 22)
    assert abs(c.export_snapshot().x_hat[0] - 22) < abs(before - 22)


def test_acknowledged_closing_bump_does_not_become_the_requested_command():
    c = MpcV2Controller(MpcV2Params())
    closing, _ = c.step(
        1000, 23, 21.5, 10, room_reported_at=1000, applied_input_managed=True
    )
    assert closing == pytest.approx(0, abs=0.001)
    c.record_applied_u(1001, 0.45)
    pending, _ = c.step(
        1030, 23, 21.5, 10, room_reported_at=1000, applied_input_managed=True
    )
    assert pending == closing
    assert c.export_snapshot().applied_u == 0.45

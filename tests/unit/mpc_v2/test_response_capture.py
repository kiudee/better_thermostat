"""Verify command capture and shared persistence at their real call sites."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from custom_components.better_thermostat.adapters.delegate import set_valve
from custom_components.better_thermostat.utils.calibration.mpc_v2 import MpcV2Params
from custom_components.better_thermostat.utils.calibration.mpc_v2.response import (
    ResponseObservation,
)
from custom_components.better_thermostat.utils.state_manager import (
    MpcV2StateData,
    StateManager,
    _deserialize,
)


async def test_successful_commands_are_recorded_between_reports(hass, monkeypatch):
    monkeypatch.setattr(
        "custom_components.better_thermostat.utils.state_manager.time", lambda: 100000
    )
    monkeypatch.setattr(
        "custom_components.better_thermostat.adapters.delegate.time", lambda: 100030
    )
    manager = StateManager(hass, "room")
    learner = manager.get_response_learner("sensor.room|weather.home|climate.valve")
    learner.observe(ResponseObservation(100000, 100000, 21, 10, 0))
    quirk = SimpleNamespace(override_set_valve=AsyncMock(return_value=True))
    trv = SimpleNamespace(
        model_quirks=quirk,
        last_valve_percent=0,
        valve_position_entity=None,
        valve_position_writable=False,
    )
    host = SimpleNamespace(
        state_mgr=manager, real_trvs={"climate.valve": trv}, device_name="Room"
    )
    assert await set_valve(host, "climate.valve", 40)
    assert learner.inputs[-1] == [100030, 40, 10]
    assert len(learner.reports) == 1
    quirk.override_set_valve.return_value = False
    assert not await set_valve(host, "climate.valve", 75)
    assert learner.inputs[-1][1] == 40


async def test_shared_evidence_saves_once_and_preserves_legacy_targets(
    hass, monkeypatch
):
    monkeypatch.setattr(
        "custom_components.better_thermostat.utils.state_manager.time", lambda: 100000
    )
    manager = StateManager(hass, "room")
    source = "sensor.room|weather.home|climate.valve"
    legacy = {"v": 1, "samples": {"35": [[1, 2, 0.05, 0.002]]}}
    manager.state.mpc_v2["eco"] = MpcV2StateData(response=legacy)
    eco = manager.get_mpc_v2_live("eco", MpcV2Params())
    comfort = manager.get_mpc_v2_live("comfort", MpcV2Params())
    eco.response = comfort.response = manager.get_response_learner(source)
    eco.response_shared = comfort.response_shared = True
    eco.response.observe(ResponseObservation(100000, 100000, 21, 10, 0))
    storage = AsyncMock()
    manager._store = storage
    await manager.save()
    saved = storage.async_save.call_args.args[0]
    assert saved["mpc_v2"]["eco"]["response"] == legacy
    assert list(saved["response_learners"]) == [source]
    assert "reports" not in saved["response_learners"][source]
    restored = StateManager(hass, "room")
    restored._state = _deserialize(saved)
    assert restored.get_response_learner(source).reports == []

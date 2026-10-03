"""Verify command capture and shared persistence at their real call sites."""

import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from homeassistant.exceptions import HomeAssistantError
import pytest

from custom_components.better_thermostat.adapters.delegate import set_valve
from custom_components.better_thermostat.trv import Trv
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
    learner.observe(ResponseObservation(100001, 100001, 21, 10, 0))
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


@pytest.mark.parametrize("refused", [None, "opening", "closing", "transient"])
async def test_deferred_close_records_only_completed_device_writes(
    hass, monkeypatch, refused
):
    quirk = importlib.import_module(
        "custom_components.better_thermostat.model_fixes.TRVZB"
    )
    clock = [100030.0]
    monkeypatch.setattr(
        "custom_components.better_thermostat.utils.state_manager.time", lambda: 100000
    )
    monkeypatch.setattr(
        "custom_components.better_thermostat.adapters.delegate.time", lambda: clock[0]
    )
    monkeypatch.setattr(
        "custom_components.better_thermostat.utils.valve_commands.time",
        lambda: clock[0],
    )
    monkeypatch.setattr(quirk, "_TRVZB_CLOSE_BUMP_DELAY_S", 0.0)
    monkeypatch.setattr(quirk, "_TRVZB_VALVE_RETRY_DELAYS_S", (0.0, 0.0), raising=False)
    manager = StateManager(hass, "deferred")
    learner = manager.get_response_learner("sensor.room|weather.home|climate.valve")
    learner.observe(ResponseObservation(100001, 100001, 21, 10, 40))
    registry = MagicMock()
    registry.async_get.return_value = SimpleNamespace(device_id="radiator")
    registry.entities.values.return_value = [
        SimpleNamespace(
            device_id="radiator", domain="number", translation_key=key, entity_id=entity
        )
        for key, entity in (
            ("valve_opening_degree", "number.opening"),
            ("valve_closing_degree", "number.closing"),
        )
    ]
    monkeypatch.setattr(quirk.er, "async_get", lambda _: registry)
    delivered = []
    refused_entity = "number.closing" if refused == "closing" else "number.opening"
    refused_value = 70 if refused == "closing" else 30
    failures = []

    async def write(_domain, _service, data, **_kwargs):
        if (
            refused
            and data["entity_id"] == refused_entity
            and data["value"] == refused_value
            and (refused != "transient" or not failures)
        ):
            failures.append(data["value"])
            raise HomeAssistantError("Simulated unacknowledged close")
        delivered.append((data["entity_id"], data["value"]))

    trv = Trv(entity_id="climate.valve", model="TRVZB", model_quirks=quirk)
    trv.last_valve_percent = 40
    host = SimpleNamespace(
        real_trvs={"climate.valve": trv},
        state_mgr=manager,
        context=None,
        in_maintenance=False,
        device_name="Room",
        hass=SimpleNamespace(
            services=SimpleNamespace(async_call=AsyncMock(side_effect=write)),
            async_create_background_task=lambda coro, name=None: asyncio.create_task(
                coro, name=name
            ),
        ),
    )
    assert await set_valve(host, "climate.valve", 30)
    task = trv.extra["_trvzb_valve_bump_task"]
    try:
        assert learner.inputs[-1][1] == 50
        assert trv.last_valve_percent == 50
        clock[0] = 100035.0
        await task
        if refused == "transient":
            assert trv.last_valve_percent == 30
            assert not trv.valve_command_uncertain
            assert len(failures) == 1
            assert ("number.opening", 30) in delivered
            assert ("number.closing", 70) in delivered
            assert not learner.reports
        elif refused:
            assert trv.last_valve_percent is None
            assert learner.diagnostics()["status"] == "valve_command_uncertain"
            assert not learner.reports
            assert (refused_entity, refused_value) not in delivered
            assert len(failures) == 3
        else:
            assert learner.inputs[-2][1] == 50
            assert learner.inputs[-1][1] == 30
            assert trv.last_valve_percent == 30
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

"""Exercise real event subscriptions and window/delivery learning guards."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

from custom_components.better_thermostat.trv import Trv
from custom_components.better_thermostat.utils.response_learning import (
    attach_response_collection,
)
from custom_components.better_thermostat.utils.state_manager import StateManager


async def test_collection_preserves_window_recovery_and_uncertain_delivery(
    hass, freezer
):
    freezer.move_to(datetime(2026, 10, 3, 8, 0, tzinfo=UTC))
    for entity, state, attrs in (
        ("sensor.room", "21", {}),
        ("weather.home", "cloudy", {"temperature": 10}),
        ("climate.valve", "heat", {}),
        ("binary_sensor.window", "off", {}),
    ):
        hass.states.async_set(entity, state, attrs)
    trv = Trv(entity_id="climate.valve", last_valve_percent=20)
    callbacks = []
    host = SimpleNamespace(
        sensor_entity_id="sensor.room",
        weather_entity="weather.home",
        outdoor_sensor=None,
        window_id="binary_sensor.window",
        door_id=None,
        real_trvs={"climate.valve": trv},
        state_mgr=StateManager(hass, "collection"),
        hass=hass,
        in_maintenance=False,
        async_on_remove=callbacks.append,
        schedule_save_state=Mock(),
    )
    learner = attach_response_collection(host, "climate.valve")
    assert learner.status == "window_recovery"
    assert not learner.reports
    freezer.tick(timedelta(minutes=30))
    hass.states.async_set("sensor.room", "21.1")
    await hass.async_block_till_done()
    assert not learner.reports
    freezer.tick(timedelta(minutes=16))
    hass.states.async_set("sensor.room", "21.2")
    await hass.async_block_till_done()
    assert len(learner.reports) == 1
    # An unchanged device heartbeat is a real report, not a controller tick.
    freezer.tick(timedelta(minutes=15))
    hass.states.async_set("sensor.room", "21.2")
    await hass.async_block_till_done()
    assert len(learner.reports) == 2
    trv.valve_command_uncertain = True
    freezer.tick(timedelta(minutes=15))
    hass.states.async_set("sensor.room", "21.3")
    await hass.async_block_till_done()
    assert learner.status == "valve_command_uncertain"
    assert not learner.reports
    assert not learner.periods
    trv.valve_command_uncertain = False
    trv.last_valve_percent = 10
    freezer.tick(timedelta(minutes=15))
    hass.states.async_set("weather.home", "cloudy", {"temperature": 10.1})
    hass.states.async_set("sensor.room", "21.4")
    await hass.async_block_till_done()
    assert len(learner.reports) == 1
    assert host.schedule_save_state.called
    for unsubscribe in callbacks:
        unsubscribe()

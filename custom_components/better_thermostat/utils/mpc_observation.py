"""Feed physical reports to MPC independently of delayed thermostat updates."""

from homeassistant.core import callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_state_report_event,
)

from .response_learning import _temperature


def physical_mpc_key(host):
    """Keep room and radiator state across comfort, eco and manual targets."""
    uid = getattr(host, "unique_id", None) or getattr(host, "_unique_id", "bt")
    return "|".join(
        (
            "mpc2",
            str(uid),
            str(getattr(host, "sensor_entity_id", None)),
            str(host.outdoor_sensor or host.weather_entity),
            *sorted(host.real_trvs),
        )
    )


def attach_mpc_observation(host, key, state):
    """Subscribe once per physical source and return the current raw report."""
    sensor_id = getattr(host, "sensor_entity_id", None)
    if host.hass is None or sensor_id is None:
        return host.cur_temp, None, False
    outdoor_id = host.outdoor_sensor or host.weather_entity
    subscriptions = getattr(host, "_mpc_observation_subscriptions", None)
    if subscriptions is None:
        subscriptions = host._mpc_observation_subscriptions = {}
    if key not in subscriptions:

        @callback
        def collect(event):
            controller = state.controller
            if controller is None:
                return
            report = event.data.get("new_state")
            if report is None:
                return
            if report.entity_id == sensor_id:
                temperature = _temperature(report)
                if temperature is not None and not report.attributes.get("restored"):
                    controller.observe_report(
                        report.last_reported.timestamp(),
                        temperature,
                        learn_disturbance=not getattr(host, "contact_open", False),
                    )
            else:
                temperature = _temperature(
                    report, weather=not bool(host.outdoor_sensor)
                )
                if temperature is not None:
                    controller.record_outdoor(
                        report.last_updated.timestamp(), temperature
                    )
            host.state_mgr.mark_dirty()
            host.schedule_save_state()
            queue = getattr(host, "control_queue_task", None)
            if queue is not None:
                if getattr(host, "in_maintenance", False):
                    host._control_needed_after_maintenance = True
                elif queue.empty():
                    queue.put_nowait(host)

        ids = [sensor_id, *([outdoor_id] if outdoor_id else [])]
        host.async_on_remove(async_track_state_change_event(host.hass, ids, collect))
        host.async_on_remove(
            async_track_state_report_event(host.hass, sensor_id, collect)
        )
        subscriptions[key] = collect
    room = host.hass.states.get(sensor_id)
    temperature = _temperature(room)
    if room is None or room.attributes.get("restored"):
        return None, None, True
    return temperature, room.last_reported.timestamp(), True

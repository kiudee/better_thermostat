"""Collect raw reports and actuator inputs independently of MPC targets."""

from __future__ import annotations

import math
from time import time

from homeassistant.core import callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_state_report_event,
)

from .calibration.mpc_v2.response import ResponseObservation
from .calibration.mpc_v2.response_model import fit_episode


def _temperature(state, *, weather=False):
    if state is None or state.state in ("unknown", "unavailable"):
        return None
    try:
        value = float(state.attributes.get("temperature") if weather else state.state)
        unit = state.attributes.get(
            "temperature_unit" if weather else "unit_of_measurement"
        )
        if unit == "°F":
            value = (value - 32) * 5 / 9
        return value if math.isfinite(value) else None
    except TypeError, ValueError:
        return None


def attach_response_collection(host, entity_id):
    """Own one collector per physical source, across all target temperatures."""
    outdoor_id = host.outdoor_sensor or host.weather_entity
    source = "|".join((str(host.sensor_entity_id), str(outdoor_id), entity_id))
    learner = host.state_mgr.get_response_learner(source)
    subscriptions = getattr(host, "_response_subscriptions", None)
    if subscriptions is None:
        subscriptions = host._response_subscriptions = {}
    if source in subscriptions:
        subscriptions[source]()
        return learner
    learner.defer_fitting = True
    contacts = []
    for value in (getattr(host, "window_id", None), getattr(host, "door_id", None)):
        contacts.extend(
            value if isinstance(value, list) else ([value] if value else [])
        )
    ids = [v for v in (host.sensor_entity_id, outdoor_id, entity_id, *contacts) if v]
    fitting = False

    async def finish_fits():
        nonlocal fitting
        try:
            while learner.pending_fits:
                reports, inputs = learner.pending_fits.popleft()
                result = await host.hass.async_add_executor_job(
                    fit_episode, reports, inputs
                )
                learner.finish_fit(reports, result)
                host.state_mgr.mark_dirty()
                host.schedule_save_state()
        finally:
            fitting = False

    @callback
    def collect(_event=None):
        nonlocal fitting
        now = time()
        room = host.hass.states.get(host.sensor_entity_id)
        outdoor = host.hass.states.get(outdoor_id) if outdoor_id else None
        valve = host.hass.states.get(entity_id)
        raw = _temperature(room)
        ambient = _temperature(outdoor, weather=not bool(host.outdoor_sensor))
        command = host.real_trvs[entity_id].last_valve_percent
        reason = None
        if raw is None:
            reason = "sensor_unavailable"
        elif (
            ambient is None
            or outdoor is None
            or not 0 <= now - outdoor.last_updated.timestamp() <= 90 * 60
        ):
            reason = "outdoor_unavailable"
        elif valve is None or valve.state in ("unavailable", "unknown"):
            reason = "valve_unavailable"
        elif host.real_trvs[entity_id].valve_command_uncertain:
            reason = "valve_command_uncertain"
        elif command is None:
            reason = "unknown_valve_command"
        elif getattr(host, "in_maintenance", False):
            reason = "maintenance"
        for contact in contacts:
            state = host.hass.states.get(contact)
            if state is None or state.state not in ("off", "closed", "false"):
                reason = "window_open_or_unavailable"
                break
            if now - state.last_changed.timestamp() < 45 * 60:
                reason = "window_recovery"
        if reason is not None:
            learner.interrupt(now, reason)
        else:
            assert room is not None and raw is not None and ambient is not None
            learner.observe(
                ResponseObservation(
                    now, room.last_reported.timestamp(), raw, ambient, float(command)
                )
            )
        host.state_mgr.mark_dirty()
        if learner.pending_fits and not fitting:
            fitting = True
            host._spawn_owned(finish_fits(), name="bt_response_fit")

    # HA sends unchanged reports on a separate event from changed states.
    host.async_on_remove(async_track_state_change_event(host.hass, ids, collect))
    host.async_on_remove(
        async_track_state_report_event(host.hass, host.sensor_entity_id, collect)
    )
    subscriptions[source] = collect
    collect()
    return learner

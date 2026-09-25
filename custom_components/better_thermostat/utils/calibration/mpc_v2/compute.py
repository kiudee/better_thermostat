"""MPC v2 entry point: one control cycle from input to percent recommendation."""

from __future__ import annotations

from dataclasses import replace
import logging
import math
from time import time

import numpy as np

from .controller import MpcV2Controller
from .io import MpcV2Input, MpcV2Output
from .params import MpcV2Params
from .response import ResponseObservation
from .state import MpcV2State, _plant_signature_of, plant_signature_differs

_LOGGER = logging.getLogger(__name__)

# When the user has no outdoor sensor we fall back to this value (°C) so the
# QP and DOB still have a defined operating point. A warm-ish German winter
# day average — close enough that the steady-state input is still in the
# valid range; well off-target temps make ``u_ss`` saturate, which the
# reference governor catches.
OUTDOOR_TEMP_FALLBACK_C = 10.0


def _all_finite(*values: float | None) -> bool:
    """Return ``True`` when every non-None value passes ``math.isfinite``."""
    for v in values:
        if v is None:
            continue
        if not math.isfinite(v):
            return False
    return True


def compute_mpc_v2(
    inp: MpcV2Input,
    params: MpcV2Params,
    state: MpcV2State | None = None,
    *,
    now: float | None = None,
) -> tuple[MpcV2Output | None, MpcV2State]:
    """Run one v2 cycle and return a percent recommendation + updated state.

    Early-exits to ``(None, state)`` when essential inputs are missing — the
    caller treats this as "hold last value".

    ``now`` overrides the wall-clock used as the controller's ``t_s``.
    Production callers leave it ``None`` (real ``time.time()``); tests
    pass a synthetic value so realistic dt-driven behaviour (DOB)
    can be exercised without sleeping.
    """
    if now is None:
        now = time()
    if state is None:
        state = MpcV2State()
    if state.created_ts == 0.0:
        state.created_ts = now

    if (
        inp.current_temp_C is None
        or inp.target_temp_C is None
        or not inp.heating_allowed
        or inp.window_open
    ):
        state.response.interrupt(now, "heating_interrupted")
        return None, state

    # Reject non-finite sensor inputs. Without this guard a NaN propagates
    # through Kalman/QP and poisons the cached state — a single bad reading
    # would require restarting the integration to recover.
    if not _all_finite(
        inp.current_temp_C, inp.target_temp_C, inp.outdoor_temp_C, inp.trv_temp_C
    ):
        state.response.interrupt(now, "invalid_input")
        _LOGGER.warning(
            "better_thermostat %s: MPC v2 (%s) non-finite input "
            "(current=%s target=%s outdoor=%s trv=%s) — holding last command",
            inp.bt_name or "BT",
            inp.entity_id or inp.key,
            inp.current_temp_C,
            inp.target_temp_C,
            inp.outdoor_temp_C,
            inp.trv_temp_C,
        )
        return None, state

    cap = max(
        0.0,
        min(100.0, inp.max_opening_pct if inp.max_opening_pct is not None else 100.0),
    )
    curve = None
    if params.learn_valve_response:
        if inp.response_source is not None:
            state.response.bind_source(inp.response_source, now)
        valid = (
            inp.learning_valid
            and inp.outdoor_temp_C is not None
            and inp.room_reported_at is not None
            and inp.outdoor_reported_at is not None
            and 0 <= now - inp.outdoor_reported_at <= 90 * 60
            and inp.applied_valve_pct is not None
        )
        state.response.observe(
            ResponseObservation(
                now=now,
                reported_at=inp.room_reported_at or 0,
                temperature=inp.current_temp_C,
                outdoor=inp.outdoor_temp_C if inp.outdoor_temp_C is not None else 10,
                valve_pct=inp.applied_valve_pct
                if inp.applied_valve_pct is not None
                else 0,
                valid=valid,
            )
        )
        learned = state.response.control_curve(prior_heat=0.05)
        if learned is not None:
            curve, loss = learned
            params = replace(params, response_heat_max=curve[-1], response_loss=loss)
    cap_u = (
        cap / 100
        if curve is None
        else float(np.interp(cap, state.response.GRID, curve)) / curve[-1]
    )
    params = replace(
        params,
        qp=replace(params.qp, u_max=cap_u),
        governor=replace(params.governor, u_max=cap_u),
    )

    new_signature = _plant_signature_of(params)
    if (
        state.controller is not None
        and state.plant_signature is not None
        and plant_signature_differs(state.plant_signature, new_signature)
    ):
        _LOGGER.info(
            "MPC v2 plant prior changed for %s (%s → %s); rebuilding controller",
            inp.key,
            state.plant_signature,
            new_signature,
        )
        state.controller = None

    if state.controller is None:
        state.controller = MpcV2Controller(params)
        state.plant_signature = new_signature

    if params.response_heat_max is not None and params.response_loss is not None:
        state.controller.update_heat_response(
            params.response_heat_max, params.response_loss
        )
    state.controller.optimiser.params.u_max = cap_u
    state.controller.governor.params.u_max = cap_u
    # Use the last successfully dispatched command, not a short adapter bump.
    applied = (
        inp.applied_valve_pct
        if inp.applied_valve_pct is not None
        else state.last_percent
    )
    if applied is not None:
        applied_u = (
            applied / 100
            if curve is None
            else float(np.interp(applied, state.response.GRID, curve)) / curve[-1]
        )
        state.controller.set_applied_u(min(cap_u, applied_u))

    if inp.outdoor_temp_C is None:
        T_outdoor = OUTDOOR_TEMP_FALLBACK_C
        if not state.outdoor_fallback_logged:
            _LOGGER.warning(
                "better_thermostat %s: MPC v2 (%s) no outdoor_temp_C — falling "
                "back to %.1f °C. Configure an outdoor sensor for accurate "
                "feed-forward (u_ss).",
                inp.bt_name or "BT",
                inp.entity_id or inp.key,
                T_outdoor,
            )
            state.outdoor_fallback_logged = True
    else:
        T_outdoor = inp.outdoor_temp_C

    u, diag = state.controller.step(
        t_s=now,
        T_room_C=inp.current_temp_C,
        T_target_C=inp.target_temp_C,
        T_outdoor_C=T_outdoor,
        T_rad_C=inp.trv_temp_C,
    )

    percent_int = round(max(0.0, min(1.0, u)) * 100.0)
    if curve is not None:
        # Select the smallest opening on plateaus, avoiding np.interp's last
        # duplicate behavior which would otherwise choose 100% at saturation.
        unique = [(curve[0], 0)]
        for pct, heat in zip(state.response.GRID[1:], curve[1:]):
            if heat > unique[-1][0] + 1e-9:
                unique.append((heat, pct))
        percent_int = round(
            float(
                np.interp(u * curve[-1], [p[0] for p in unique], [p[1] for p in unique])
            )
        )
    if inp.max_opening_pct is not None:
        # The cap is a percent by contract; clamp it into 0..100 here so an
        # out-of-range value from a caller cannot widen or invert the limit.
        percent_int = min(percent_int, int(max(0.0, min(100.0, inp.max_opening_pct))))

    # Feed the actually-applied (possibly capped) fraction back so the observer
    # and rate limiter track the real valve input, not the uncapped request.
    if state.controller is not None:
        actual_u = (
            percent_int / 100
            if curve is None
            else float(np.interp(percent_int, state.response.GRID, curve)) / curve[-1]
        )
        state.controller.set_applied_u(actual_u)

    if params.learn_valve_response:
        diag.response_curve = state.response.diagnostics()
        diag.response_curve["control_active"] = curve is not None

    state.last_percent = float(percent_int)
    state.last_compute_ts = now

    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug(
            "better_thermostat %s: MPC v2 (%s) target=%.2f current=%.2f trv=%s "
            "outdoor=%s -> valve=%d%% (T_rad_hat=%s D_hat=%.4f tau_room=%.0f) key=%s",
            inp.bt_name or "BT",
            inp.entity_id or inp.key,
            inp.target_temp_C,
            inp.current_temp_C,
            inp.trv_temp_C,
            inp.outdoor_temp_C,
            percent_int,
            diag.T_rad_hat,
            diag.D_hat_K_per_min,
            diag.tau_room_min,
            inp.key,
        )

    return MpcV2Output(valve_percent=percent_int, diagnostics=diag), state

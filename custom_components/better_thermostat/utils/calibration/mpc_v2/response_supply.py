"""Observation-only shared valve shape with independently varying heat supply.

All shape, loss and lag choices use training periods only. The later period
calibrates its heat scale on a bounded prefix, then predicts the remaining tail.
Neither a fit nor its reported saturation authorizes heating control.
"""

from importlib import import_module
from itertools import product
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .response_model import GRID, _design, period_exposure

type Profile = tuple[float, float, float, float]


def _shape(opening: float, deadzone: float, knee: float) -> float:
    return max(0.0, min(1.0, (opening - deadzone) / (knee - deadzone)))


def _matrix(
    period: dict[str, Any], profile: Profile
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    deadzone, knee, lag, loss = profile
    inputs = [
        [t, 100 * _shape(u, deadzone, knee), outdoor]
        for t, u, outdoor in period["inputs"]
    ]
    return _design(period["reports"], inputs, [100.0], loss, lag)


def _fit(
    matrix: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    require_heat: bool = True,
) -> NDArray[np.float64] | None:
    columns = [0, 1, 2] if np.linalg.norm(matrix[:, 0]) > 1e-8 else [1, 2]
    if require_heat and len(columns) != 3:
        return None
    x = matrix[:, columns]
    if len(values) < len(columns):
        return None
    scales = np.linalg.norm(x, axis=0)
    if np.any(scales < 1e-8):
        return None
    normalized = x / scales
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular[-1] / singular[0] < 0.002:
        return None
    lower = np.array([0, 0, -0.03])[columns]
    upper = np.array([0.2, 0.2, 0.03])[columns]
    daqp = import_module("daqp")
    solution, _, flag, _ = daqp.solve(
        np.ascontiguousarray(
            2 * normalized.T @ normalized + np.eye(len(columns)) * 1e-12
        ),
        np.ascontiguousarray(-2 * normalized.T @ values),
        np.eye(len(columns)),
        upper * scales,
        lower * scales,
        primal_tol=1e-9,
        dual_tol=1e-9,
    )
    if flag != 1 or not np.all(np.isfinite(solution)):
        return None
    result = np.zeros(3)
    result[columns] = solution / scales
    return result


def fit_variable_supply(
    periods: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Learn a bounded relative response only after independent tail prediction."""
    if len(periods) < 4:
        return None, "waiting_for_supply_periods"
    exposure = [period_exposure(p["reports"], p["inputs"]) for p in periods]
    heated = [
        i
        for i, (_, minutes, maximum) in enumerate(exposure)
        if minutes >= 5 and maximum >= 10
    ]
    if len(heated) < 4:
        return None, "waiting_for_supply_periods"
    validation_index = heated[-1]
    training, validation = periods[:validation_index], periods[validation_index]
    maximum = max(e[2] for e in exposure[:validation_index])
    if exposure[validation_index][2] > maximum:
        return None, "waiting_for_matching_openings"
    # A single opening per period cannot separate curve shape from heat supply.
    varied = 0
    for period in training:
        durations: dict[float, float] = {}
        end = period["reports"][-1][0]
        for a, b in zip(period["inputs"], [*period["inputs"][1:], [end, 0, 0]]):
            if a[1] >= 5:
                durations[a[1]] = (
                    durations.get(a[1], 0) + max(0, min(end, b[0]) - a[0]) / 60
                )
        varied += sum(minutes >= 5 for minutes in durations.values()) >= 2
    if varied < 2:
        return None, "unidentified_valve_shape"
    profiles = product(
        (0.0, 5.0, 10.0, 15.0, 20.0),
        sorted({min(k, maximum) for k in (25.0, 35.0, 50.0, maximum)}),
        (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 30.0, 45.0),
        np.geomspace(0.0003, 0.006, 12),
    )
    candidates: list[tuple[float, Profile, list[float]]] = []
    for deadzone, knee, lag, loss in profiles:
        if knee <= deadzone:
            continue
        profile = (deadzone, knee, lag, float(loss))
        residuals: list[float] = []
        gains: list[float] = []
        for period in training:
            matrix, values = _matrix(period, profile)
            solution = _fit(matrix, values, require_heat=False)
            if solution is None:
                break
            error = matrix @ solution - values
            if max(abs(error)) > 0.35:
                break
            residuals.extend(error)
            if np.linalg.norm(matrix[:, 0]) > 1e-8:
                gains.append(float(solution[0]))
        else:
            rmse = float(np.sqrt(np.mean(np.square(residuals))))
            if rmse <= 0.10 and len(gains) >= 3 and min(gains) >= 0.003:
                candidates.append((rmse, profile, gains))
    if not candidates:
        return None, "supply_model_training_mismatch"
    best = min(candidates, key=lambda c: c[0])
    _, profile, gains = best
    matrix, values = _matrix(validation, profile)
    # Choose prefix length by excitation and rank, before looking at its errors.
    prefix = None
    solution = None
    # Three unknowns fitted to three reports interpolate sensor noise. Require
    # redundant evidence and enough heating to distinguish supply from carryover.
    for count in range(5, min(12, len(values) // 2) + 1):
        calibration = {
            "reports": validation["reports"][: count + 1],
            "inputs": validation["inputs"],
        }
        if period_exposure(calibration["reports"], calibration["inputs"])[1] < 30:
            continue
        solution = _fit(matrix[:count], values[:count])
        if solution is not None:
            prefix = count
            break
    if prefix is None or solution is None:
        return None, "waiting_for_supply_calibration"
    remaining_minutes = (
        validation["reports"][-1][0] - validation["reports"][prefix][0]
    ) / 60
    future_heat = period_exposure(validation["reports"][prefix:], validation["inputs"])[
        1
    ]
    # A cooling tail can validate retained heat, but cannot validate valve gain.
    if remaining_minutes < 45 or future_heat < 15:
        return None, "insufficient_supply_validation"
    residual = matrix @ solution - values
    future = residual[prefix:]
    rmse = float(np.sqrt(np.mean(future**2)))
    flat = (
        np.array([r[1] for r in validation["reports"][prefix + 1 :]])
        - validation["reports"][prefix][1]
    )
    no_heat = matrix.copy()
    no_heat[:, 0] = 0
    baseline = _fit(no_heat[:prefix], values[:prefix], require_heat=False)
    baseline_rmse = (
        math.inf
        if baseline is None
        else float(
            np.sqrt(np.mean((no_heat[prefix:] @ baseline - values[prefix:]) ** 2))
        )
    )
    if (
        max(abs(residual[:prefix])) > 0.35
        or rmse > 0.15
        or max(abs(future)) > 0.35
        or rmse > 0.8 * float(np.sqrt(np.mean(flat**2)))
        or rmse > 0.8 * baseline_rmse
    ):
        return None, "supply_model_validation_mismatch"
    deadzone, knee, lag, loss = profile
    plausible = [c for c in candidates if c[0] <= best[0] + 0.02]
    relative: list[float | None] = []
    heat: list[float | None] = []
    lower: list[float | None] = []
    upper: list[float | None] = []
    for opening in GRID:
        if opening > maximum:
            relative.append(None)
            heat.append(None)
            lower.append(None)
            upper.append(None)
            continue
        shapes = [_shape(opening, c[1][0], c[1][1]) for c in plausible]
        if max(shapes) - min(shapes) > 0.35:
            relative.append(None)
            heat.append(None)
            lower.append(None)
            upper.append(None)
            continue
        shape = _shape(opening, deadzone, knee)
        relative.append(shape)
        heat.append(float(np.median(gains)) * shape)
        lower.append(
            max(0.0, min(min(c[2]) * s for c, s in zip(plausible, shapes)) - 0.003)
        )
        upper.append(max(max(c[2]) * s for c, s in zip(plausible, shapes)) + 0.003)
    if not any(v is not None and v >= 0.003 for v in heat[1:]):
        return None, "unidentified_valve_shape"
    return {
        "start": periods[0]["reports"][0][0],
        "end": validation["reports"][-1][0],
        "reports": sum(len(p["reports"]) for p in periods[: validation_index + 1]),
        "maximum_opening": maximum,
        "heat": heat,
        "lower": lower,
        "upper": upper,
        "loss": loss,
        "validation_rmse": rmse,
        "validation_kind": "separate_period",
        "period_count": validation_index + 1,
        "model": "variable_supply",
        "relative_heat": relative,
        "supply_range": [min(gains), max(gains)],
        "radiator_lag_min": lag,
        "validation_prefix_end": validation["reports"][prefix][0],
        "validation_max_error": float(max(abs(future))),
        "train_rmse": best[0],
    }, "candidate_observed"

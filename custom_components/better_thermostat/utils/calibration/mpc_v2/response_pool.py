"""Validate shared heating response across separate clean periods."""

from __future__ import annotations

from importlib import import_module
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .response_model import GRID, _basis, _bounded_fit, _design, period_exposure


def _joint_design(
    periods: list[dict[str, Any]], knots: list[float], loss: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    width = len(knots) + 2 * len(periods)
    matrices: list[NDArray[np.float64]] = []
    values: list[NDArray[np.float64]] = []
    for index, period in enumerate(periods):
        matrix, value = _design(period["reports"], period["inputs"], knots, loss)
        joint = np.zeros((len(value), width))
        joint[:, : len(knots)] = matrix[:, : len(knots)]
        begin = len(knots) + 2 * index
        joint[:, begin : begin + 2] = matrix[:, -2:]
        matrices.append(joint)
        values.append(value)
    return np.vstack(matrices), np.concatenate(values)


def _joint_fit(
    matrix: NDArray[np.float64], values: NDArray[np.float64], count: int
) -> tuple[NDArray[np.float64], float] | None:
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale < 1e-6) or len(values) < matrix.shape[1] + 2:
        return None
    normalized = matrix / scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    if singular[-1] / singular[0] < 0.002:
        return None
    width = matrix.shape[1]
    lower = np.zeros(width)
    upper = np.full(width, 0.2)
    lower[count + 1 :: 2] = -0.03
    upper[count + 1 :: 2] = 0.03
    bounds = np.vstack(
        (np.eye(width), np.r_[1 / scale[:count], np.zeros(width - count)])
    )
    solver = import_module("daqp")
    solution, _objective, flag, _info = solver.solve(
        np.ascontiguousarray(2 * normalized.T @ normalized + np.eye(width) * 1e-12),
        np.ascontiguousarray(-2 * normalized.T @ values),
        np.ascontiguousarray(bounds),
        np.r_[upper * scale, 0.2],
        np.r_[lower * scale, -math.inf],
        primal_tol=1e-9,
        dual_tol=1e-9,
        iter_limit=1000,
    )
    if flag != 1 or not np.all(np.isfinite(solution)):
        return None
    solution /= scale
    residual = matrix @ solution - values
    if np.any(solution[:count] < -1e-7) or max(abs(residual)) > 0.35:
        return None
    return solution, float(np.mean(residual**2))


def _validate(
    period: dict[str, Any],
    knots: list[float],
    loss: float,
    response: NDArray[np.float64],
) -> float | None:
    matrix, values = _design(period["reports"], period["inputs"], knots, loss)
    known = matrix[:, : len(knots)] @ response
    nuisance = _bounded_fit(matrix[:2, -2:], values[:2] - known[:2])
    baseline = _bounded_fit(matrix[:2, -2:], values[:2])
    if nuisance is None or baseline is None:
        return None
    residual = known + matrix[:, -2:] @ nuisance[1] - values
    validation = residual[2:]
    rmse = float(np.sqrt(np.mean(validation**2)))
    no_heat = matrix[:, -2:] @ baseline[1] - values
    zero_rmse = float(np.sqrt(np.mean(no_heat[2:] ** 2)))
    flat_rmse = float(
        np.sqrt(
            np.mean(
                (
                    np.array([r[1] for r in period["reports"][3:]])
                    - period["reports"][2][1]
                )
                ** 2
            )
        )
    )
    if (
        rmse > 0.15
        or max(abs(validation)) > 0.35
        or rmse > 0.8 * zero_rmse
        or rmse > 0.8 * flat_rmse
    ):
        return None
    return rmse


def fit_short_periods(
    periods: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Fit on earlier periods and predict a separate heated period."""
    if len(periods) < 3:
        return None, "waiting_for_independent_periods"
    exposure = [period_exposure(p["reports"], p["inputs"]) for p in periods]
    heated_indices = [
        i
        for i, (_, heat, maximum) in enumerate(exposure)
        if heat >= 5 and maximum >= 10
    ]
    if (
        len(heated_indices) < 3
        or sum(e[1] for e in exposure) < 60
        or sum(e[0] for e in exposure) < 45
    ):
        return None, "waiting_for_heating_evidence"
    validation_index = heated_indices[-1]
    training = periods[:validation_index]
    validation = periods[validation_index]
    training_max = max(e[2] for e in exposure[:validation_index])
    if exposure[validation_index][2] > training_max:
        return None, "waiting_for_matching_openings"
    knots = sorted({min(15.0, training_max), min(35.0, training_max), training_max})
    candidates: list[dict[str, Any]] = []
    for loss in np.geomspace(0.0003, 0.006, 16):
        matrix, values = _joint_design(training, knots, loss)
        fit = _joint_fit(matrix, values, len(knots))
        if fit is None:
            continue
        solution, error = fit
        if math.sqrt(error) > 0.10:
            continue
        rmse = _validate(validation, knots, loss, solution[: len(knots)])
        if rmse is None:
            continue
        covariance = np.linalg.pinv(matrix.T @ matrix) * max(0.03**2, error)
        curve: list[float | None] = []
        sensitivity: list[float] = []
        for opening in GRID:
            direction = np.zeros(matrix.shape[1])
            direction[: len(knots)] = _basis(opening, knots)
            curve.append(
                float(direction @ solution) if opening <= training_max else None
            )
            sensitivity.append(
                2 * math.sqrt(max(0.0, float(direction @ covariance @ direction)))
            )
        candidates.append(
            {
                "loss": float(loss),
                "curve": curve,
                "sensitivity": sensitivity,
                "error": error,
                "rmse": rmse,
            }
        )
    if not candidates:
        return None, "inconsistent_or_unidentifiable_model"
    best = min(candidates, key=lambda c: c["error"])
    plausible = [
        c
        for c in candidates
        if math.sqrt(c["error"]) <= math.sqrt(best["error"]) + 0.02
    ]
    lower: list[float | None] = []
    upper: list[float | None] = []
    for index, center in enumerate(best["curve"]):
        if center is None:
            lower.append(None)
            upper.append(None)
            continue
        lo = max(
            0.0,
            min(c["curve"][index] - c["sensitivity"][index] for c in plausible) - 0.003,
        )
        hi = max(c["curve"][index] + c["sensitivity"][index] for c in plausible) + 0.003
        if hi - lo > 0.06:
            best["curve"][index] = None
            lower.append(None)
            upper.append(None)
        else:
            lower.append(lo)
            upper.append(hi)
    if not any(v is not None and v >= 0.003 for v in best["curve"][1:]):
        return None, "inconsistent_or_unidentifiable_model"
    used = periods[: validation_index + 1]
    return {
        "start": used[0]["reports"][0][0],
        "end": validation["reports"][-1][0],
        "reports": sum(len(p["reports"]) for p in used),
        "maximum_opening": training_max,
        "heat": best["curve"],
        "lower": lower,
        "upper": upper,
        "loss": best["loss"],
        "validation_rmse": best["rmse"],
        "period_count": len(used),
        "validation_kind": "separate_period",
    }, "candidate_observed"

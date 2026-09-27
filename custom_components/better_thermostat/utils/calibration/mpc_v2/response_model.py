"""Fit sparse temperature reports using the complete command history.

This is an observation-only model. Its envelope describes sensitivity to
plausible loss coefficients, not a calibrated statistical confidence interval.
"""

from __future__ import annotations

from itertools import product
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

GRID = tuple(range(0, 101, 5))
RADIATOR_LAG_MIN = 15.0


def _basis(opening: float, knots: list[float]) -> NDArray[np.float64]:
    left = np.array([0.0, *knots[:-1]])
    return np.clip((opening - left) / (np.array(knots) - left), 0, 1)


def _design(
    reports: list[list[float]],
    inputs: list[list[float]],
    knots: list[float],
    loss: float,
):
    """Integrate the linear thermal states exactly between input changes."""
    count = len(knots) + 2  # response increments, initial radiator heat, disturbance
    room = np.zeros(count)
    radiator = np.zeros(count)
    radiator[-2] = 1.0
    base = reports[0][1]
    previous = reports[0][0]
    index = 0
    rows: list[NDArray[np.float64]] = []
    values: list[float] = []
    for timestamp, temperature in reports[1:]:
        while previous < timestamp:
            while index + 1 < len(inputs) and inputs[index + 1][0] <= previous:
                index += 1
            end = min(
                timestamp,
                inputs[index + 1][0] if index + 1 < len(inputs) else timestamp,
            )
            dt = (end - previous) / 60
            _, opening, outdoor = inputs[index]
            er = math.exp(-loss * dt)
            eh = math.exp(-dt / RADIATOR_LAG_MIN)
            retained = (er - eh) / (1 / RADIATOR_LAG_MIN - loss)
            sustained = (1 - er) / loss
            basis = np.zeros(count)
            basis[: len(knots)] = _basis(opening, knots)
            room = er * room + retained * radiator + (sustained - retained) * basis
            room[-1] += sustained
            radiator = eh * radiator + (1 - eh) * basis
            base = er * base + (1 - er) * outdoor
            previous = end
        rows.append(room.copy())
        values.append(temperature - base)
    return np.array(rows), np.array(values)


def _bounded_fit(matrix: NDArray[np.float64], values: NDArray[np.float64]):
    """Enumerate the small nonnegative active set without a solver dependency."""
    best = None
    for active in product((False, True), repeat=matrix.shape[1] - 1):
        columns = [i for i, enabled in enumerate(active) if enabled] + [
            matrix.shape[1] - 1
        ]
        solution = np.zeros(matrix.shape[1])
        solution[columns] = np.linalg.lstsq(matrix[:, columns], values, rcond=None)[0]
        if (
            np.any(solution[:-1] < -1e-9)
            or sum(solution[:-2]) > 0.2
            or solution[-2] > 0.2
            or abs(solution[-1]) > 0.03
        ):
            continue
        solution[:-1] = np.maximum(0, solution[:-1])
        error = float(np.mean((matrix @ solution - values) ** 2))
        if best is None or error < best[0]:
            best = (error, solution)
    return best


def fit_episode(
    reports: list[list[float]], inputs: list[list[float]]
) -> tuple[dict[str, Any] | None, str]:
    """Return a candidate only when varied inputs identify a validated model."""
    if len(reports) < 12 or len(inputs) < 2:
        return None, "insufficient_reports"
    durations = []
    for a, b in zip(inputs, [*inputs[1:], [reports[-1][0], 0, 0]]):
        durations.append((a[1], max(0, b[0] - a[0]) / 60))
    off = sum(dt for opening, dt in durations if opening <= 1)
    heated = sum(dt for opening, dt in durations if opening >= 5)
    maximum = max(opening for opening, dt in durations if dt > 0)
    if off < 45 or heated < 60 or maximum < 10:
        return None, "insufficient_excitation"
    knots = sorted({min(15.0, maximum), min(35.0, maximum), maximum})
    train_end = max(len(knots) + 4, int((len(reports) - 1) * 0.75))
    candidates: list[dict[str, Any]] = []
    for loss in np.geomspace(0.0003, 0.006, 16):
        matrix, values = _design(reports, inputs, knots, loss)
        train = matrix[:train_end]
        scale = np.linalg.norm(train, axis=0)
        if np.any(scale < 1e-6):
            continue
        singular = np.linalg.svd(train / scale, compute_uv=False)
        if singular[-1] / singular[0] < 0.002:
            continue
        fit = _bounded_fit(train, values[:train_end])
        if fit is None:
            continue
        error, solution = fit
        residual = matrix @ solution - values
        validation = residual[train_end:]
        rmse = float(np.sqrt(np.mean(validation**2))) if len(validation) else math.inf
        # A constant-temperature predictor is a useful independent floor.
        baseline = float(
            np.sqrt(
                np.mean(
                    (
                        np.array([r[1] for r in reports[train_end + 1 :]])
                        - reports[train_end][1]
                    )
                    ** 2
                )
            )
        )
        if (
            math.sqrt(error) > 0.10
            or rmse > 0.15
            or max(abs(residual)) > 0.35
            or rmse > baseline * 0.8
        ):
            continue
        curve = [
            float(_basis(p, knots) @ solution[: len(knots)]) if p <= maximum else None
            for p in GRID
        ]
        covariance = np.linalg.pinv(train.T @ train) * max(0.03**2, error)
        sensitivity = []
        for opening in GRID:
            direction = np.zeros(matrix.shape[1])
            direction[: len(knots)] = _basis(opening, knots)
            sensitivity.append(
                2 * math.sqrt(max(0.0, float(direction @ covariance @ direction)))
            )
        candidates.append(
            {
                "loss": float(loss),
                "curve": curve,
                "rmse": rmse,
                "train_rmse": math.sqrt(error),
                "sensitivity": sensitivity,
            }
        )
    if not candidates:
        return None, "inconsistent_or_unidentifiable_model"
    best = min(candidates, key=lambda c: c["train_rmse"])
    plausible = [c for c in candidates if c["train_rmse"] <= best["train_rmse"] + 0.02]
    low: list[float | None] = []
    high: list[float | None] = []
    for i, center in enumerate(best["curve"]):
        if center is None:
            low.append(None)
            high.append(None)
        else:
            lo = max(
                0.0, min(c["curve"][i] - c["sensitivity"][i] for c in plausible) - 0.003
            )
            hi = max(c["curve"][i] + c["sensitivity"][i] for c in plausible) + 0.003
            if hi - lo > 0.06:
                best["curve"][i] = None
                low.append(None)
                high.append(None)
            else:
                low.append(lo)
                high.append(hi)
    if not any(v is not None and v >= 0.003 for v in best["curve"][1:]):
        return None, "insufficient_excitation"
    return {
        "start": reports[0][0],
        "end": reports[-1][0],
        "reports": len(reports),
        "maximum_opening": maximum,
        "heat": best["curve"],
        "lower": low,
        "upper": high,
        "loss": best["loss"],
        "validation_rmse": best["rmse"],
    }, "candidate_observed"

"""Conservative identification of delivered room heat from settled valve holds.

The learned quantity is K/min of room heating, not hydraulic flow. Each
independent hold contributes one observation. Repeated controller ticks and
overlapping slope windows never increase the confidence sample count.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from statistics import median
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class ResponseObservation:
    """A timestamped room report and the command held before that report."""

    now: float
    reported_at: float
    temperature: float
    outdoor: float
    valve_pct: float
    valid: bool = True


class ValveResponseLearner:
    """Learn only after settling, with fresh reports and a nearby off baseline.

    Unknown regions retain a physical prior for control and null confidence
    bounds for display. No probes or autonomous cap changes are performed.
    """

    VERSION = 1
    SETTLE_S = 30 * 60.0
    OFF_SETTLE_S = 45 * 60.0
    WINDOW_S = 15 * 60.0
    MAX_GAP_S = 10 * 60.0
    BASELINE_AGE_S = 2 * 60 * 60.0
    RETENTION_S = 14 * 24 * 60 * 60.0
    GRID = tuple(range(0, 101, 5))

    def __init__(self) -> None:
        """Start without evidence or a contiguous measurement segment."""
        self.samples: dict[int, list[list[float]]] = {}
        self.baseline: tuple[float, float, float] | None = None
        self.status = "waiting_for_reports"
        self.rejected = 0
        self._points: deque[ResponseObservation] = deque(maxlen=256)
        self._last_report = -math.inf
        self._last_tick = -math.inf
        self._hold_start = 0.0
        self._command: float | None = None
        self._now = 0.0
        self.source: str | None = None

    def bind_source(self, source: str, now: float) -> None:
        """Do not transfer a calibration to a different sensor/valve pairing."""
        if source != self.source:
            self.samples.clear()
            self.interrupt(now, "source_changed")
            self.source = source

    def interrupt(self, now: float, reason: str) -> None:
        """Discard an interrupted interval without erasing learned evidence."""
        self._points.clear()
        self._command = None
        self.baseline = None
        self._hold_start = now
        self.status = reason

    def observe(self, obs: ResponseObservation) -> None:
        """Accept a fresh report or reject the entire questionable interval."""
        values = (obs.now, obs.reported_at, obs.temperature, obs.outdoor, obs.valve_pct)
        if not all(math.isfinite(v) for v in values):
            self.interrupt(self._now, "invalid_input")
            self.rejected += 1
            return
        self._now = obs.now
        if (
            not obs.valid
            or not -10 <= obs.temperature <= 45
            or not -50 <= obs.outdoor <= 50
            or not 0 <= obs.valve_pct <= 100
            or not 0 <= obs.now - obs.reported_at <= self.MAX_GAP_S
            or obs.now <= self._last_tick
        ):
            self.interrupt(obs.now, "unreliable_input")
            self.rejected += 1
            return
        if obs.now - self._last_tick > self.MAX_GAP_S:
            self.interrupt(obs.now, "report_gap")
        self._last_tick = obs.now
        if self._command is None or abs(obs.valve_pct - self._command) > 1:
            self._command = obs.valve_pct
            self._hold_start = obs.now
            self._points.clear()
        if obs.reported_at <= self._last_report:
            self.status = "waiting_for_fresh_report"
            return
        if obs.reported_at - self._last_report > self.MAX_GAP_S:
            self.interrupt(obs.now, "report_gap")
            self._command = obs.valve_pct
        self._last_report = obs.reported_at
        settling = self.OFF_SETTLE_S if obs.valve_pct <= 1 else self.SETTLE_S
        if obs.now - self._hold_start < settling:
            self.status = "settling"
            return
        if self._points:
            last = self._points[-1]
            dt_min = (obs.reported_at - last.reported_at) / 60
            if abs(obs.temperature - last.temperature) > max(0.4, 0.12 * dt_min):
                self.interrupt(obs.now, "temperature_jump")
                self.rejected += 1
                return
        self._points.append(obs)
        while (
            self._points
            and obs.reported_at - self._points[0].reported_at > 2 * self.WINDOW_S
        ):
            self._points.popleft()
        points = list(self._points)
        if (
            len(points) < 4
            or points[-1].reported_at - points[0].reported_at < self.WINDOW_S
        ):
            self.status = "collecting_window"
            return
        if max(p.outdoor for p in points) - min(p.outdoor for p in points) > 1:
            self.status = "outdoor_change"
            return
        # Theil–Sen slope reduces the influence of one noisy or quantized report.
        slopes = [
            (b.temperature - a.temperature) / ((b.reported_at - a.reported_at) / 60)
            for i, a in enumerate(points)
            for b in points[i + 1 :]
            if b.reported_at - a.reported_at >= 5 * 60
        ]
        slope = median(slopes)
        intercept = median(
            p.temperature - slope * (p.reported_at - points[0].reported_at) / 60
            for p in points
        )
        residual = max(
            abs(
                p.temperature
                - intercept
                - slope * (p.reported_at - points[0].reported_at) / 60
            )
            for p in points
        )
        if residual > 0.15 or abs(slope) > 0.12:
            self.status = "inconsistent_slope"
            self.rejected += 1
            return
        delta = median(p.temperature - p.outdoor for p in points)
        if obs.valve_pct <= 1:
            if delta >= 4 and -0.06 <= slope <= -0.001:
                self.baseline = (obs.now, -slope / delta, obs.outdoor)
                self.status = "off_baseline"
            else:
                self.status = "ambiguous_off_baseline"
            return
        if self.baseline is None or obs.now - self.baseline[0] > self.BASELINE_AGE_S:
            self.status = "needs_recent_off_baseline"
            return
        if abs(obs.outdoor - self.baseline[2]) > 2:
            self.status = "baseline_weather_changed"
            return
        heat = slope + self.baseline[1] * delta
        if not 0.003 <= heat <= 0.2:
            self.status = "no_identifiable_heat"
            return
        bucket = int(5 * round(obs.valve_pct / 5))
        if bucket == 0:
            self.status = "below_resolution"
            return
        samples = self.samples.setdefault(bucket, [])
        entry = [self._hold_start, obs.now, heat, self.baseline[1]]
        if samples and samples[-1][0] == self._hold_start:
            samples[-1] = entry
        else:
            samples.append(entry)
        self.samples[bucket] = [
            s for s in samples if obs.now - s[1] <= self.RETENTION_S
        ][-32:]
        self.status = "learning"

    def _retained(self, bucket: int) -> list[list[float]]:
        return [
            s
            for s in self.samples.get(bucket, [])
            if 0 <= self._now - s[1] <= self.RETENTION_S
        ]

    @staticmethod
    def _interval(values: list[float]) -> tuple[float | None, float | None]:
        # Distribution-free, at least 95% confidence interval for the median
        # of independent holds. Six holds are needed for finite endpoints.
        n = len(values)
        if n < 6:
            return None, None
        k = 0
        for candidate in range(1, n // 2 + 1):
            tail = 2 * sum(math.comb(n, j) for j in range(candidate)) / 2**n
            if tail <= 0.05:
                k = candidate
        ordered = sorted(values)
        floor = 0.003  # Do not turn repeatable quantization into false precision.
        return max(0, ordered[k - 1] - floor), ordered[n - k] + floor

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe point estimates, confidence bounds and coverage."""
        means: list[float | None] = []
        lower: list[float | None] = []
        upper: list[float | None] = []
        counts = []
        for bucket in self.GRID:
            values = [s[2] for s in self._retained(bucket)]
            counts.append(len(values))
            means.append(median(values) if values else (0.0 if bucket == 0 else None))
            lo, hi = self._interval(values)
            lower.append(lo)
            upper.append(hi)
        saturation = None
        # Evidence must reach 100%, with repeated intermediate high-opening
        # holds. Overlapping wide intervals are not evidence of equivalence.
        supported = [i for i in range(1, len(self.GRID)) if lower[i] is not None]
        control = self.control_curve(prior_heat=0.05)
        if control is not None and supported and supported[-1] == len(self.GRID) - 1:
            for i in supported:
                tail = [j for j in supported if j >= i]
                if len(tail) < 3 or max(b - a for a, b in zip(tail, tail[1:])) > 4:
                    continue
                if all(
                    0
                    <= cast(float, upper[j]) - cast(float, lower[i])
                    <= 0.15 * cast(float, means[-1])
                    for j in tail
                ):
                    saturation = self.GRID[i]
                    break
        return {
            "opening_pct": list(self.GRID),
            "heat_K_min": means,
            "lower_K_min": lower,
            "upper_K_min": upper,
            "independent_holds": counts,
            "saturation_pct": saturation,
            "status": self.status,
            "rejected_windows": self.rejected,
            "control_heat_K_min": control[0] if control is not None else None,
        }

    def control_curve(self, prior_heat: float) -> tuple[list[float], float] | None:
        """Return a cautious monotone heat curve and local loss coefficient.

        Three independent holds are required before influencing control. The
        prior is retained outside observed coverage; it is never displayed as
        measured evidence. A higher heat estimate makes MPC brake earlier.
        """
        points = [(0.0, 0.0)]
        losses = []
        for bucket in self.GRID[1:]:
            samples = self._retained(bucket)
            if len(samples) >= 3:
                values = [s[2] for s in samples]
                center = median(values)
                if max(values) - min(values) > max(0.015, center * 0.5):
                    continue
                points.append((float(bucket), center))
                losses.extend(s[3] for s in samples)
        if len(points) < 2:
            return None
        # Substantially less heat at larger openings indicates a changed
        # boiler/room regime, not a valve curve that should be inverted.
        high = 0.0
        for _, heat in points:
            if heat < high - max(0.006, high * 0.15):
                return None
            high = max(high, heat)
        # Beyond the highest observed point, retain increasing prior demand.
        if points[-1][0] < 100:
            x, y = points[-1]
            points.append((100.0, y + prior_heat * (1 - x / 100)))
        # The monotone upper envelope avoids making a low, confounded hold
        # erase stronger heating evidence at a smaller opening.
        ys = np.maximum.accumulate([y for _, y in points])
        curve = [
            float(v)
            for v in np.atleast_1d(np.interp(self.GRID, [x for x, _ in points], ys))
        ]
        return curve, median(losses)

    def export(self) -> dict[str, Any]:
        """Persist evidence only; never resume an unfinished hold after restart."""
        return {
            "v": self.VERSION,
            "source": self.source,
            "samples": {str(k): self._retained(k) for k in self.samples},
        }

    def restore(self, payload: object, now: float) -> None:
        """Validate bounded, finite evidence and ignore malformed snapshots."""
        self._now = now
        if not isinstance(payload, dict) or payload.get("v") != self.VERSION:
            return
        raw = payload.get("samples")
        if not isinstance(raw, dict):
            return
        source = payload.get("source")
        self.source = source if isinstance(source, str) and len(source) <= 512 else None
        for bucket in self.GRID[1:]:
            values = raw.get(str(bucket), [])
            if not isinstance(values, list):
                continue
            good = []
            seen = set()
            for item in values[-32:]:
                if not isinstance(item, list) or len(item) != 4:
                    continue
                if not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in item
                ):
                    continue
                start, end, heat, loss = item
                if (
                    0 <= now - end <= self.RETENTION_S
                    and 0 <= start <= end
                    and start not in seen
                    and 0.003 <= heat <= 0.2
                    and 0 < loss <= 0.015
                ):
                    good.append(item)
                    seen.add(start)
            self.samples[bucket] = good
        self.status = "restored_waiting_for_reports"

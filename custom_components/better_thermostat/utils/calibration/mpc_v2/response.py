"""Observation-only valve response learning from irregular real reports."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import math
from statistics import median
from typing import Any

from .response_model import GRID, fit_episode


@dataclass(frozen=True)
class ResponseObservation:
    """An actual room report together with the currently applied input."""

    now: float
    reported_at: float
    temperature: float
    outdoor: float
    valve_pct: float
    valid: bool = True
    reason: str = "unreliable_input"


class ValveResponseLearner:
    """Accumulate independent episodes; never count silence as a measurement."""

    VERSION = 2
    GRID = GRID
    RETENTION_S = 14 * 86400.0
    MAX_EPISODES = 24
    MAX_REPORTS = 1024
    MAX_INPUTS = 2048
    EPISODE_S = 6 * 3600.0
    MAX_EPISODE_S = 24 * 3600.0

    def __init__(self) -> None:
        """Start with no assumed temperature or response evidence."""
        self.source: str | None = None
        self.status = "waiting_for_report"
        self.episodes: list[dict[str, Any]] = []
        self.rejections: Counter[str] = Counter()
        self.recent: deque[dict[str, Any]] = deque(maxlen=12)
        self.reports: list[list[float]] = []
        self.inputs: list[list[float]] = []
        self._now = 0.0
        self._last_report = -math.inf
        self._not_before = 0.0
        self._last_command: float | None = None
        self._last_outdoor: float | None = None
        self._gaps: deque[float] = deque(maxlen=24)
        self.legacy: dict[str, Any] | None = None
        self.defer_fitting = False
        self.pending_fits: deque[tuple[list[list[float]], list[list[float]]]] = deque(
            maxlen=2
        )

    @property
    def heartbeat_limit_s(self) -> float:
        """Allow normal hourly silence, with a bounded cadence allowance."""
        return max(90 * 60.0, min(150 * 60.0, 1.75 * max(self._gaps, default=0)))

    def bind_source(self, source: str, now: float) -> None:
        """Prevent transferring evidence to a different physical source."""
        if self.source != source:
            self.episodes.clear()
            self.pending_fits.clear()
            self._gaps.clear()
            self._last_report = -math.inf
            self.interrupt(now, "source_changed")
            self.source = source

    def interrupt(self, now: float, reason: str) -> None:
        """Discard an invalid pending episode, retaining completed evidence."""
        if self.reports:
            self.rejections[reason] += 1
            self.recent.append(
                {
                    "start": self.reports[0][0],
                    "end": now,
                    "reason": reason,
                    "reports": len(self.reports),
                }
            )
        self.reports.clear()
        self.inputs.clear()
        self._not_before = max(self._not_before, now)
        self.status = reason
        self._now = max(self._now, now)

    def record_command(self, now: float, percent: float) -> None:
        """Record a successful actuator command at its actual write time."""
        if (
            not math.isfinite(now)
            or not math.isfinite(percent)
            or not 0 <= percent <= 100
            or now < self._now
        ):
            return
        self._last_command = percent
        self._now = now
        if self.reports and self._last_outdoor is not None:
            self._input(now, percent, self._last_outdoor)

    def _input(self, now: float, percent: float, outdoor: float) -> None:
        row = [now, percent, outdoor]
        if self.inputs and now == self.inputs[-1][0]:
            self.inputs[-1] = row
        elif not self.inputs or row[1:] != self.inputs[-1][1:]:
            self.inputs.append(row)
        if len(self.inputs) > self.MAX_INPUTS:
            self.interrupt(now, "input_history_limit")

    def observe(self, obs: ResponseObservation) -> None:
        """Update inputs every tick, and measurements only on a new report."""
        if not all(
            math.isfinite(v)
            for v in (
                obs.now,
                obs.reported_at,
                obs.temperature,
                obs.outdoor,
                obs.valve_pct,
            )
        ):
            self.interrupt(self._now, "invalid_input")
            return
        if obs.now < self._now:
            return
        self._now = obs.now
        self.episodes = [
            e for e in self.episodes if 0 <= obs.now - e["end"] <= self.RETENTION_S
        ]
        if not obs.valid:
            self.interrupt(obs.now, obs.reason)
            return
        if not (
            -10 <= obs.temperature <= 45
            and -50 <= obs.outdoor <= 50
            and 0 <= obs.valve_pct <= 100
            and obs.reported_at <= obs.now
        ):
            self.interrupt(obs.now, "invalid_input")
            return
        self._last_command, self._last_outdoor = obs.valve_pct, obs.outdoor
        if obs.reported_at < self._last_report:
            return
        if obs.now - obs.reported_at > self.heartbeat_limit_s:
            self.interrupt(obs.now, "missing_heartbeat")
            return
        if obs.now < self._not_before or obs.reported_at < self._not_before:
            self.status = "recovery"
            return
        if self.reports:
            self._input(obs.now, obs.valve_pct, obs.outdoor)
        if obs.reported_at <= self._last_report:
            self.status = "waiting_for_report"
            return
        if self.reports:
            previous, temperature = self.reports[-1]
            gap = obs.reported_at - previous
            if gap > self.heartbeat_limit_s:
                self.interrupt(obs.now, "missing_heartbeat")
                return
            if abs(obs.temperature - temperature) > max(0.4, 0.12 * gap / 60):
                self.interrupt(obs.now, "temperature_jump")
                self._not_before = obs.now + 45 * 60
                return
            if gap > 0:
                self._gaps.append(gap)
        self._last_report = obs.reported_at
        if not self.reports:
            self.inputs = [[obs.reported_at, obs.valve_pct, obs.outdoor]]
        self.reports.append([obs.reported_at, obs.temperature])
        self.status = "collecting_episode"
        duration = self.reports[-1][0] - self.reports[0][0]
        if len(self.reports) >= 12 and duration >= self.EPISODE_S:
            reports, inputs = self.reports, self.inputs
            if self.defer_fitting:
                self.pending_fits.append((reports, inputs))
                self.status = "evaluating_episode"
            else:
                self.finish_fit(reports, fit_episode(reports, inputs))
            self.reports = [self.reports[-1]]
            self.inputs = [[obs.reported_at, obs.valve_pct, obs.outdoor]]
        elif duration > self.MAX_EPISODE_S or len(self.reports) >= self.MAX_REPORTS:
            self.interrupt(obs.now, "insufficient_reports")

    def finish_fit(
        self, reports: list[list[float]], result: tuple[dict[str, Any] | None, str]
    ) -> None:
        """Commit a pure model fit on the owning event loop."""
        evidence, reason = result
        self.recent.append(
            {
                "start": reports[0][0],
                "end": reports[-1][0],
                "reason": reason,
                "reports": len(reports),
            }
        )
        if evidence is not None:
            self.episodes = [*self.episodes, evidence][-self.MAX_EPISODES :]
        else:
            self.rejections[reason] += 1
        self.status = reason

    def diagnostics(self) -> dict[str, Any]:
        """Expose candidate coverage and sensitivity without implying control."""
        self.episodes = [
            e for e in self.episodes if 0 <= self._now - e["end"] <= self.RETENTION_S
        ]
        heat: list[float | None] = []
        lower: list[float | None] = []
        upper: list[float | None] = []
        counts: list[int] = []
        for i, _opening in enumerate(self.GRID):
            support = [e for e in self.episodes if e["heat"][i] is not None]
            counts.append(len(support))
            heat.append(median(e["heat"][i] for e in support) if support else None)
            lower.append(min(e["lower"][i] for e in support) if support else None)
            upper.append(max(e["upper"][i] for e in support) if support else None)
        consistent = len(self.episodes) >= 3 and any(
            n >= 3 and hi - lo < max(0.015, mid * 0.5)
            for n, lo, hi, mid in zip(counts[1:], lower[1:], upper[1:], heat[1:])
            if mid is not None and hi is not None and lo is not None
        )
        return {
            "version": self.VERSION,
            "mode": "observation_only",
            "opening_pct": list(self.GRID),
            "heat_K_min": heat,
            "lower_K_min": lower,
            "upper_K_min": upper,
            "independent_holds": counts,
            "episode_coverage": counts,
            "accepted_episodes": len(self.episodes),
            "candidate_consistent": consistent,
            "saturation_pct": None,
            "status": self.status,
            "control_heat_K_min": None,
            "control_active": False,
            "rejected_windows": sum(self.rejections.values()),
            "rejection_reasons": dict(self.rejections),
            "pending_reports": len(self.reports),
            "pending_commands": len(self.inputs),
            "pending_duration_min": (self._now - self.reports[0][0]) / 60
            if self.reports
            else 0,
            "report_age_min": (self._now - self._last_report) / 60
            if math.isfinite(self._last_report)
            else None,
            "heartbeat_limit_min": self.heartbeat_limit_s / 60,
            "computed_at": self._now,
            "recent_episodes": list(self.recent),
            "validation_rmse_C": max(
                (e["validation_rmse"] for e in self.episodes), default=None
            ),
            "uncertainty_kind": "model_sensitivity_and_episode_range",
        }

    def control_curve(self, prior_heat: float) -> None:
        """Candidates remain observation-only throughout this release."""
        return None

    def export(self) -> dict[str, Any]:
        """Persist completed evidence, never an unfinished interval."""
        return {
            "v": self.VERSION,
            "source": self.source,
            "episodes": self.episodes,
            "rejections": dict(self.rejections),
            "recent": list(self.recent),
            "gaps": list(self._gaps),
            "legacy": self.legacy,
        }

    def restore(self, payload: object, now: float) -> None:
        """Validate bounded evidence; legacy holds are archived, not reused."""
        self._now = now
        if not isinstance(payload, dict):
            return
        source = payload.get("source")
        self.source = source if isinstance(source, str) and len(source) <= 512 else None
        if payload.get("v") == 1:
            samples = payload.get("samples", {})
            if isinstance(samples, dict):
                clean = {
                    str(k): [
                        s
                        for s in v[-32:]
                        if isinstance(s, list)
                        and len(s) == 4
                        and all(
                            isinstance(n, (float, int)) and math.isfinite(n) for n in s
                        )
                    ]
                    for k, v in list(samples.items())[:20]
                    if str(k) in {str(p) for p in self.GRID[1:]} and isinstance(v, list)
                }
                self.legacy = {"v": 1, "source": self.source, "samples": clean}
            return
        if payload.get("v") != self.VERSION:
            return
        episodes = payload.get("episodes", [])
        for e in episodes[-self.MAX_EPISODES :] if isinstance(episodes, list) else []:
            if not isinstance(e, dict):
                continue
            try:
                if not (
                    0 <= now - e["end"] <= self.RETENTION_S
                    and 0 <= e["start"] < e["end"]
                    and 12 <= e["reports"] <= self.MAX_REPORTS
                    and 0 < e["loss"] < 0.02
                    and 0 <= e["validation_rmse"] <= 0.15
                    and 10 <= e["maximum_opening"] <= 100
                ):
                    continue
                if any(
                    (mid is None and (lo is not None or hi is not None))
                    or (
                        mid is not None
                        and (lo is None or hi is None or not lo <= mid <= hi)
                    )
                    for mid, lo, hi in zip(e["heat"], e["lower"], e["upper"])
                ):
                    continue
                if any(
                    len(e[k]) != len(self.GRID)
                    or any(
                        v is not None
                        and (
                            not isinstance(v, (float, int))
                            or not math.isfinite(v)
                            or not 0 <= v <= 0.25
                        )
                        for v in e[k]
                    )
                    for k in ("heat", "lower", "upper")
                ):
                    continue
                if any(
                    x["start"] < e["end"] and e["start"] < x["end"]
                    for x in self.episodes
                ):
                    continue
                self.episodes.append(
                    {
                        k: e[k]
                        for k in (
                            "start",
                            "end",
                            "reports",
                            "loss",
                            "validation_rmse",
                            "maximum_opening",
                            "heat",
                            "lower",
                            "upper",
                        )
                    }
                )
            except KeyError, TypeError, ValueError:
                continue
        gaps = payload.get("gaps", [])
        for gap in gaps[-24:] if isinstance(gaps, list) else []:
            if (
                isinstance(gap, (float, int))
                and math.isfinite(gap)
                and 0 < gap <= 150 * 60
            ):
                self._gaps.append(gap)
        reasons = payload.get("rejections", {})
        if isinstance(reasons, dict):
            self.rejections.update(
                {
                    k: min(v, 1000000)
                    for k, v in list(reasons.items())[:32]
                    if isinstance(k, str)
                    and len(k) < 80
                    and isinstance(v, int)
                    and v >= 0
                }
            )
        recent = payload.get("recent", [])
        for item in recent[-12:] if isinstance(recent, list) else []:
            if (
                isinstance(item, dict)
                and isinstance(item.get("reason"), str)
                and len(item["reason"]) < 80
            ):
                numbers = [item.get(k) for k in ("start", "end", "reports")]
                if all(
                    isinstance(n, (int, float)) and math.isfinite(n) and n >= 0
                    for n in numbers
                ):
                    self.recent.append(
                        {k: item[k] for k in ("start", "end", "reports", "reason")}
                    )
        legacy = payload.get("legacy")
        if isinstance(legacy, dict) and legacy.get("v") == 1:
            archived = ValveResponseLearner()
            archived.restore(legacy, now)
            self.legacy = archived.legacy
        self.status = "restored_waiting_for_report"

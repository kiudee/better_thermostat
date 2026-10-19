"""Explicit assumed valve characteristic, separate from learned room heating."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ValveFlowPrior:
    """Map physical opening percent to a relative flow input and back."""

    onset_pct: float = 19.5
    near_full_pct: float = 40.0
    shape: float = 1.3

    def __post_init__(self) -> None:
        """Reject invalid parameters before they can reach the controller."""
        if not (
            all(math.isfinite(v) for v in self.signature)
            and 0 <= self.onset_pct < self.near_full_pct <= 100
            and 1 <= self.shape <= 4
        ):
            raise ValueError("Invalid valve flow prior")

    @property
    def signature(self) -> tuple[float, float, float]:
        """Identify the input units used by a controller snapshot."""
        return (self.onset_pct, self.near_full_pct, self.shape)

    @property
    def identity(self) -> str:
        """Return a stable identity for persistence and diagnostics."""
        return "weibull:" + ":".join(format(v, ".12g") for v in self.signature)

    def flow(self, opening_pct: float) -> float:
        """Return assumed relative flow for an acknowledged physical command."""
        if not math.isfinite(opening_pct):
            raise ValueError("Non-finite valve opening")
        x = max(0.0, min(100.0, opening_pct) - self.onset_pct)
        x /= self.near_full_pct - self.onset_pct
        return -math.expm1(-math.log(50) * x**self.shape)

    def command(self, flow: float, cap_pct: float = 100) -> int:
        """Choose the closest available flow, with an explicit physical shutoff.

        Compare quantized flow rather than rounding the continuous opening:
        a tiny solver residual must not become 20% opening and real heating.
        """
        if not math.isfinite(flow) or not math.isfinite(cap_pct):
            raise ValueError("Non-finite flow request or opening limit")
        cap = int(max(0.0, min(100.0, cap_pct)))
        target = min(max(0.0, flow), self.flow(cap))
        if target <= 0:
            return 0
        if target >= self.flow(cap):
            opening = float(cap)
        else:
            opening = self.onset_pct + (self.near_full_pct - self.onset_pct) * (
                -math.log1p(-target) / math.log(50)
            ) ** (1 / self.shape)
        candidates = {0, min(cap, math.floor(opening)), min(cap, math.ceil(opening))}
        return min(candidates, key=lambda pct: (abs(self.flow(pct) - target), pct))

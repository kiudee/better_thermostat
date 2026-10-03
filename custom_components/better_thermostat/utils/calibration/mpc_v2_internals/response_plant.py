"""Room temperature plus retained radiator heat, driven by learned heat input."""

from __future__ import annotations

from typing import override

import numpy as np

from ._types import FloatArray
from .plant import PlantModelRC2, PlantParams


class HeatResponsePlant(PlantModelRC2):
    """Use K/min of delivered heat instead of a fictitious hydraulic fraction.

    State two is retained heating rate, not radiator temperature. The existing
    optimiser only observes room temperature and can use this linear model.
    """

    def __init__(self, params: PlantParams, dt_s: float, heat_max: float, loss: float):
        """Create a first-order radiator-lag model with an identified heat scale."""
        super().__init__(params, dt_s)
        self.heat_max = heat_max
        self.loss = loss

    @override
    def linearised_AB(
        self, T_outdoor_C: float, T_rad_op_C: float
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return discrete temperature/heat-rate dynamics for the optimiser."""
        dt = self.dt_min
        decay = np.exp(-dt / self.params.tau_rad_min)
        # Integrating heat over the step avoids a full extra sample of delay.
        retained = self.params.tau_rad_min * (1 - decay)
        A = np.array([[1 - dt * self.loss, retained], [0.0, decay]])
        B = np.array([[(dt - retained) * self.heat_max], [(1 - decay) * self.heat_max]])
        return A, B, np.array([dt * self.loss * T_outdoor_C, 0.0])

    @override
    def discrete_step(
        self, x: FloatArray, u: float, T_outdoor_C: float, D_K_per_min: float = 0.0
    ) -> FloatArray:
        """Advance the temperature and retained heat estimate."""
        A, B, d = self.linearised_AB(T_outdoor_C, 0)
        result = A @ x + B.flatten() * np.clip(u, 0, 1) + d
        result[0] += D_K_per_min * self.dt_min
        return result

    @override
    def steady_radiator_temp(
        self, T_setpoint_C: float, T_outdoor_C: float, D_hat_K_per_min: float = 0.0
    ) -> float:
        """Return the second state at equilibrium: required room heating rate."""
        return self.loss * (T_setpoint_C - T_outdoor_C) - D_hat_K_per_min

    @override
    def steady_input(
        self, T_setpoint_C: float, T_outdoor_C: float, D_hat_K_per_min: float = 0.0
    ) -> float:
        """Return required heat as a fraction of the learned heat scale."""
        return (
            self.steady_radiator_temp(T_setpoint_C, T_outdoor_C, D_hat_K_per_min)
            / self.heat_max
        )

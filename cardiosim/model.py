"""
Right-hand side of the 12-state closed-loop cardiovascular model.

The ODE system is defined in Sec. 2 of the manuscript. This module
provides a thin, stateless callable ``ClosedLoopCardiovascular`` that
takes parameters and initial conditions as constructor arguments and
exposes the vector field

    dx/dt = Phi(x, t)

through its ``rhs`` method, which is the function consumed by
``scipy.integrate.solve_ivp``.

The module also exposes scalar helpers for the time-varying cardiac
activation a(t), the ventricular elastances E_{L,R}(t), and the active
pressure sources P_{L,R}(t) — all of them evaluated identically in
``rhs`` and in :pyfunc:`cardiosim.postprocess`, so that waveforms and
pressure–volume loops are always internally consistent.
"""

from __future__ import annotations
import numpy as np

from cardiosim.parameters import Parameters
from cardiosim.initial_state import InitialState


class ClosedLoopCardiovascular:
    """Vector field of the 12-state closed-loop cardiovascular model.

    Parameters
    ----------
    parameters : Parameters
        Immutable parameter set.
    initial_state : InitialState
        Initial condition vector.

    Notes
    -----
    The object is intentionally *stateless after construction*: the
    ``rhs`` callable depends only on ``(t, y)`` and the frozen
    ``parameters`` attribute. No simulation data is kept on the instance.
    """

    def __init__(self, parameters: Parameters, initial_state: InitialState):
        self.p = parameters
        self.x0 = initial_state

    # ------------------------------------------------------------------ #
    # Cardiac activation and ventricular auxiliaries                     #
    # ------------------------------------------------------------------ #
    def activation(self, t: float) -> float:
        """Cardiac activation function a(t) (Eq. 5 of the manuscript)."""
        t_c = self.p.cardiac_period
        t_s = self.p.systolic_duration
        t_m = np.mod(t, t_c)
        return np.where(
            t_m < t_s,
            0.5 * (1.0 - np.cos(2.0 * np.pi * t_m / t_s)),
            0.0,
        )

    def elastance_left(self, t: float) -> float:
        """E_L(t) = E_d_L + E_s_L a(t)."""
        return self.p.E_d_left + self.p.E_s_left * self.activation(t)

    def elastance_right(self, t: float) -> float:
        """E_R(t) = E_d_R + E_s_R a(t)."""
        return self.p.E_d_right + self.p.E_s_right * self.activation(t)

    def source_pressure_left(self, t: float) -> float:
        """Active (isovolumetric-source) component P_L(t) = P0_L a(t)."""
        return self.p.P0_left * self.activation(t)

    def source_pressure_right(self, t: float) -> float:
        """Active (isovolumetric-source) component P_R(t) = P0_R a(t)."""
        return self.p.P0_right * self.activation(t)

    # ------------------------------------------------------------------ #
    # Right-hand side of the 12-dim ODE system                           #
    # ------------------------------------------------------------------ #
    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Return dx/dt evaluated at time ``t`` and state ``y``.

        Signature matches the ``fun(t, y)`` contract of
        :func:`scipy.integrate.solve_ivp`.
        """
        p = self.p

        # Unpack state for readability
        (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12) = y

        # --- Time-varying quantities shared with postprocess -----------
        P_L = self.source_pressure_left(t)
        P_R = self.source_pressure_right(t)
        E_L = self.elastance_left(t)
        E_R = self.elastance_right(t)

        # --- Valve pressure differences (Eqs. 9 and 13) ----------------
        dP1 = P_L + x12 * E_L - x1         # aortic     (LV -> aorta)
        dP2 = x5 - P_R - x6 * E_R          # tricuspid  (RA -> RV)
        dP3 = P_R + x6 * E_R - x7          # pulmonary  (RV -> PA)
        dP4 = x11 - P_L - x12 * E_L        # mitral     (LA -> LV)

        # --- Diode switches (Eqs. 8 and 12) ----------------------------
        s1 = (dP1 > 0.0) - p.aortic_reflux       # aortic valve
        s2 = 1.0 if dP2 > 0.0 else 0.0           # tricuspid
        s3 = 1.0 if dP3 > 0.0 else 0.0           # pulmonary
        s4 = 1.0 if dP4 > 0.0 else 0.0           # mitral

        # --- Inflow / outflow resistances ------------------------------
        Z_ao_out = (s1 * dP1) / (p.R_myo_left + p.R_aortic_valve)
        Z_tri_in = (s2 * dP2) / p.R_tricuspid
        Z_pul_out = (s3 * dP3) / (p.R_myo_right + p.R_pulmonary_valve)
        Z_mit_in = (s4 * dP4) / p.R_mitral

        # --- 12 coupled ODEs (Eq. 17 of the manuscript) ----------------
        dx1 = (Z_ao_out - x2) / p.C_ao
        dx2 = (x1 - p.R_ao_sys * x2 - x3) / p.L_ao_sys
        dx3 = (x2 - x4) / p.C_sys_art
        dx4 = (x3 - p.R_sys * x4 - x5) / p.L_sys
        dx5 = (x4 - Z_tri_in) / p.C_sys_ven
        dx6 = Z_tri_in - Z_pul_out
        dx7 = (Z_pul_out - x8) / p.C_pa
        dx8 = (x7 - p.R_pa * x8 - x9) / p.L_pa
        dx9 = (x8 - x10) / p.C_pul_art
        dx10 = (x9 - p.R_pul * x10 - x11) / p.L_pul
        dx11 = (x10 - Z_mit_in) / p.C_pul_ven
        dx12 = Z_mit_in - Z_ao_out

        return np.array(
            [dx1, dx2, dx3, dx4, dx5, dx6, dx7, dx8, dx9, dx10, dx11, dx12],
            dtype=float,
        )

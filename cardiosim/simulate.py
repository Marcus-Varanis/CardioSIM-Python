"""
Numerical integration of the closed-loop cardiovascular model.

This module provides :func:`run_simulation`, a thin, typed wrapper
around :func:`scipy.integrate.solve_ivp`, and :class:`SimulationResult`,
a dataclass that packages solver output together with the parameter
set, initial state, and solver metadata so that downstream modules
(post-processing, visualization) receive a self-describing object.

Design rationale
----------------
The manuscript's original implementation stored simulation results on
the plotter instance, coupling "what was computed" to "how it will be
drawn". Here the two concerns are strictly separated: numerical data
flow as immutable :class:`SimulationResult` objects; plotting functions
in :mod:`cardiosim.visualization` take such objects and return
``matplotlib`` figures without side effects.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import time

import numpy as np
from scipy.integrate import solve_ivp

from cardiosim.parameters import Parameters
from cardiosim.initial_state import InitialState
from cardiosim.model import ClosedLoopCardiovascular


SolverName = Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]


@dataclass
class SimulationResult:
    """Container for the numerical solution of the cardiovascular model.

    Attributes
    ----------
    t : np.ndarray
        Time points, shape ``(N,)``.
    y : np.ndarray
        State trajectory, shape ``(12, N)``. Row indices follow
        :class:`~cardiosim.initial_state.InitialState`.
    parameters : Parameters
        Parameters that generated this run.
    initial_state : InitialState
        Initial condition used.
    solver : str
        Name of the scipy solver employed.
    rtol, atol : float
        Tolerances used by the adaptive integrator.
    wall_time_s : float
        Wall-clock time of the integration (seconds).
    nfev : int
        Number of right-hand-side evaluations.
    nsteps : int
        Number of accepted steps.
    model : ClosedLoopCardiovascular
        The model instance (needed by post-processing to recover
        time-varying auxiliaries such as elastances).
    """

    t: np.ndarray
    y: np.ndarray
    parameters: Parameters
    initial_state: InitialState
    solver: str
    rtol: float
    atol: float
    wall_time_s: float
    nfev: int
    nsteps: int
    model: ClosedLoopCardiovascular

    # --- Convenient accessors by physiological name ------------------- #
    @property
    def P_aortic(self) -> np.ndarray:
        return self.y[0]

    @property
    def Q_sys_art(self) -> np.ndarray:
        return self.y[1]

    @property
    def P_sys(self) -> np.ndarray:
        return self.y[2]

    @property
    def Q_sys_ven(self) -> np.ndarray:
        return self.y[3]

    @property
    def P_right_atrial(self) -> np.ndarray:
        return self.y[4]

    @property
    def V_right_vent(self) -> np.ndarray:
        return self.y[5]

    @property
    def P_pul_venous(self) -> np.ndarray:
        return self.y[6]

    @property
    def Q_pul_art(self) -> np.ndarray:
        return self.y[7]

    @property
    def P_pul(self) -> np.ndarray:
        return self.y[8]

    @property
    def Q_pul_ven(self) -> np.ndarray:
        return self.y[9]

    @property
    def P_left_atrial(self) -> np.ndarray:
        return self.y[10]

    @property
    def V_left_vent(self) -> np.ndarray:
        return self.y[11]


def run_simulation(
    parameters: Parameters,
    initial_state: InitialState,
    t_end: float,
    *,
    t_start: float = 0.0,
    solver: SolverName = "RK45",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    n_eval: Optional[int] = None,
    max_step: Optional[float] = None,
) -> SimulationResult:
    """Integrate the closed-loop cardiovascular model.

    Parameters
    ----------
    parameters, initial_state : Parameters, InitialState
        Physiological inputs.
    t_end : float
        Final simulation time (s).
    t_start : float, default 0.0
        Initial simulation time (s).
    solver : str
        Any solver accepted by ``scipy.integrate.solve_ivp``.
    rtol, atol : float
        Relative and absolute tolerances. The defaults (1e-10, 1e-12)
        follow the manuscript and are safely within the envelope
        validated by the benchmark in the Supplementary Material.
    n_eval : int, optional
        If provided, the solution is sampled on a uniform grid of
        ``n_eval`` points; otherwise the solver's adaptive grid is
        returned. Uniform sampling is required for LSODA, which does
        not support ``dense_output``.
    max_step : float, optional
        Forwarded to ``solve_ivp``.

    Returns
    -------
    SimulationResult
    """
    model = ClosedLoopCardiovascular(parameters, initial_state)
    y0 = initial_state.to_array()
    t_span = (t_start, t_end)

    t_eval = None
    dense_output = True
    if n_eval is not None:
        t_eval = np.linspace(t_start, t_end, n_eval)
    if solver.upper() == "LSODA":
        # LSODA in SciPy does not support dense_output; fall back to
        # t_eval-based sampling.
        dense_output = False
        if t_eval is None:
            t_eval = np.linspace(t_start, t_end, 20000)

    kwargs = dict(
        fun=model.rhs,
        t_span=t_span,
        y0=y0,
        method=solver,
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        t_eval=t_eval,
    )
    if max_step is not None:
        kwargs["max_step"] = max_step

    t0 = time.perf_counter()
    sol = solve_ivp(**kwargs)
    wall = time.perf_counter() - t0

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return SimulationResult(
        t=sol.t,
        y=sol.y,
        parameters=parameters,
        initial_state=initial_state,
        solver=solver,
        rtol=rtol,
        atol=atol,
        wall_time_s=wall,
        nfev=int(sol.nfev),
        nsteps=int(sol.t.size),
        model=model,
    )

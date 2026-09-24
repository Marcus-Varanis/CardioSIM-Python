"""
cardiosim — Closed-loop lumped-parameter cardiovascular model.

A teaching-oriented Python framework accompanying the paper:
    "A Closed-Loop Cardiovascular Model as a Teaching Framework for
     Physics, Mathematical Modeling, and Scientific Computing"
    (Revista Brasileira de Ensino de Física).

Top-level design principles
---------------------------
1.  Immutable parameters and initial state via frozen dataclasses,
    so that every simulation is bit-for-bit reproducible.
2.  Strict separation of concerns:

        parameters.py       : physiological constants (dataclass)
        initial_state.py    : initial vector (dataclass)
        model.py            : right-hand side of the ODE system
        simulate.py         : thin, typed wrapper around scipy.solve_ivp
        postprocess.py      : derived quantities (P_LV, P_RV, metrics)
        visualization.py    : pure plotting functions (navy / gold style)

3.  Derived ventricular pressures are computed *in one place*
    (postprocess.derive_ventricular_pressures) so that time-series plots
    and pressure–volume loops always use the same definition:

        P_v(t) = P_0 a(t) + V(t) E(t) - R_m Z_out(t)

    This removes the silent discrepancy present in earlier "plotter"
    implementations that used only the active source term P_0 a(t).
"""

from cardiosim.parameters import Parameters, baseline_parameters
from cardiosim.initial_state import InitialState, baseline_initial_state
from cardiosim.model import ClosedLoopCardiovascular
from cardiosim.simulate import SimulationResult, run_simulation
from cardiosim.postprocess import (
    derive_ventricular_pressures,
    cycle_metrics,
    last_cycle_slice,
)

__version__ = "1.0.0"
__all__ = [
    "Parameters",
    "baseline_parameters",
    "InitialState",
    "baseline_initial_state",
    "ClosedLoopCardiovascular",
    "SimulationResult",
    "run_simulation",
    "derive_ventricular_pressures",
    "cycle_metrics",
    "last_cycle_slice",
]

"""
Post-processing of cardiovascular simulation results.

This module derives quantities that are **functions of the state** but
that the integrator does not carry directly — most importantly, the
intracavitary ventricular pressures that close the pressure–volume
loops.

Why this module exists
----------------------
In earlier implementations, the pressure–volume loops were drawn using
the *source* pressures ``P_0 a(t)`` only. That is not the ventricular
pressure: it omits the elastance-times-volume contribution (and, when
the outflow valve is open, the myocardial viscosity drop).

The correct expression, consistent with Eqs. (11) and (15) of the
manuscript, is

    P_vL(t) = P_L(t) + E_L(t) V_LV(t) - R_L * Z_oL(t),
    P_vR(t) = P_R(t) + E_R(t) V_RV(t) - R_R * Z_oR(t),

where ``Z_oL`` and ``Z_oR`` are the outflow terms through the aortic
and pulmonary valves respectively. :func:`derive_ventricular_pressures`
evaluates these using the **same** model instance that generated the
trajectory, so there is no risk of numerical inconsistency.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cardiosim.simulate import SimulationResult


@dataclass
class VentricularPressures:
    """Derived intracavitary ventricular pressures."""
    P_LV: np.ndarray     #: left-ventricular pressure (mmHg)
    P_RV: np.ndarray     #: right-ventricular pressure (mmHg)


def derive_ventricular_pressures(
    result: SimulationResult,
    *,
    subtract_myocardial_drop: bool = True,
) -> VentricularPressures:
    """Compute intracavitary P_LV(t) and P_RV(t) from a SimulationResult.

    Parameters
    ----------
    result : SimulationResult
        Output of :func:`cardiosim.simulate.run_simulation`.
    subtract_myocardial_drop : bool, default True
        If ``True`` (recommended), the myocardial viscosity drop
        ``R_m * Z_out`` is subtracted when the outflow valve is open,
        matching Eqs. (11) and (15) of the manuscript. Setting this to
        ``False`` returns the bare sum ``P_0 a(t) + E(t) V(t)``, which
        is sometimes used as a simplification in undergraduate
        discussions.

    Returns
    -------
    VentricularPressures
    """
    t = result.t
    m = result.model
    p = result.parameters

    # Time-varying ventricular auxiliaries, evaluated on the output grid
    a = m.activation(t)
    E_L = p.E_d_left + p.E_s_left * a
    E_R = p.E_d_right + p.E_s_right * a
    P_srcL = p.P0_left * a
    P_srcR = p.P0_right * a

    V_LV = result.V_left_vent
    V_RV = result.V_right_vent

    # Pressure-source + elastance-volume contribution
    P_LV = P_srcL + E_L * V_LV
    P_RV = P_srcR + E_R * V_RV

    if subtract_myocardial_drop:
        # Outflow terms (aortic and pulmonary valves)
        P_ao = result.P_aortic
        P_pa = result.P_pul_venous        # x_7, downstream of pulmonary valve

        dP1 = P_LV - P_ao                 # aortic valve ΔP
        dP3 = P_RV - P_pa                 # pulmonary valve ΔP

        s1 = (dP1 > 0).astype(float) - p.aortic_reflux
        s3 = (dP3 > 0).astype(float)

        Z_oL = (s1 * dP1) / (p.R_myo_left + p.R_aortic_valve)
        Z_oR = (s3 * dP3) / (p.R_myo_right + p.R_pulmonary_valve)

        P_LV = P_LV - p.R_myo_left * Z_oL
        P_RV = P_RV - p.R_myo_right * Z_oR

    return VentricularPressures(P_LV=P_LV, P_RV=P_RV)


# ---------------------------------------------------------------------- #
# Cycle-level metrics                                                    #
# ---------------------------------------------------------------------- #
def last_cycle_slice(result: SimulationResult,
                     n_cycles: int = 1) -> slice:
    """Return a ``slice`` selecting the last ``n_cycles`` cardiac cycles."""
    t = result.t
    t_c = result.parameters.cardiac_period
    t_end = t[-1]
    t_cut = t_end - n_cycles * t_c
    idx = np.searchsorted(t, t_cut, side="left")
    return slice(idx, None)


@dataclass
class CycleMetrics:
    """Summary metrics extracted from the last stable cycle."""
    P_syst_aortic: float   #: systolic aortic pressure (mmHg)
    P_diast_aortic: float  #: diastolic aortic pressure (mmHg)
    pulse_pressure: float  #: P_syst - P_diast (mmHg)
    MAP: float             #: mean arterial pressure (mmHg)
    EDV_LV: float          #: left-ventricular end-diastolic volume (mL)
    ESV_LV: float          #: left-ventricular end-systolic volume (mL)
    SV_LV: float           #: stroke volume, left ventricle (mL/beat)
    EF_LV: float           #: ejection fraction, left ventricle (%)
    HR: float              #: heart rate (bpm)
    CO: float              #: cardiac output (L/min)


def cycle_metrics(result: SimulationResult) -> CycleMetrics:
    """Extract standard hemodynamic metrics from the last stable cycle."""
    sl = last_cycle_slice(result, n_cycles=1)
    P_ao = result.P_aortic[sl]
    V_LV = result.V_left_vent[sl]

    P_syst = float(np.max(P_ao))
    P_diast = float(np.min(P_ao))
    PP = P_syst - P_diast
    MAP = P_diast + PP / 3.0

    EDV = float(np.max(V_LV))
    ESV = float(np.min(V_LV))
    SV = EDV - ESV
    EF = 100.0 * SV / EDV

    HR = 60.0 / result.parameters.cardiac_period
    CO = SV * HR / 1000.0   # mL/beat * beats/min -> L/min

    return CycleMetrics(
        P_syst_aortic=P_syst,
        P_diast_aortic=P_diast,
        pulse_pressure=PP,
        MAP=MAP,
        EDV_LV=EDV,
        ESV_LV=ESV,
        SV_LV=SV,
        EF_LV=EF,
        HR=HR,
        CO=CO,
    )

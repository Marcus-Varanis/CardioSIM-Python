"""
Physiological and circuit-analog parameters of the closed-loop model.

The naming convention follows the manuscript:

    C_i     : compliance        (mmHg^-1 · cm^3)     — capacitance analog
    L_i     : inertance         (mmHg · s^2 · cm^-3) — inductance analog
    R_i     : resistance        (mmHg · s · cm^-3)   — resistor analog
    E_s, E_d: systolic/diastolic elastance  (mmHg · cm^-3)
    P_0     : peak isovolumetric pressure            (mmHg)
    t_c     : cardiac period                         (s)

Subscript map (consistent with Fig. 3 of the manuscript):

    1 -> aortic                     4 -> pulmonary artery (arterial)
    2 -> arterial systemic          5 -> arterial pulmonary
    3 -> venous systemic            6 -> venous pulmonary

Valve resistances R_1, R_4, R_5, R_8 correspond respectively to the
aortic, tricuspid, pulmonary, and mitral valves.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class Parameters:
    """Immutable container for all cardiovascular model parameters.

    Instances are ``frozen`` so that the parameter set associated with a
    given simulation cannot be silently mutated, which is essential for
    reproducibility. Create a perturbed copy with :pyfunc:`modified`.

    Examples
    --------
    >>> p = baseline_parameters()
    >>> p_hypertensive = p.modified(R_sys=2.0 * p.R_sys)
    """

    # --- Compliances (mmHg^-1 · cm^3) -----------------------------------
    C_ao: float       #: aortic compliance                        C_1
    C_sys_art: float  #: arterial systemic compliance             C_2
    C_sys_ven: float  #: venous systemic compliance               C_3
    C_pa: float       #: pulmonary artery compliance              C_4
    C_pul_art: float  #: arterial pulmonary compliance            C_5
    C_pul_ven: float  #: venous pulmonary compliance              C_6

    # --- Inertances (mmHg · s^2 · cm^-3) --------------------------------
    L_ao_sys: float   #: aortic-systemic inertance                L_1
    L_sys: float      #: systemic inertance                       L_2
    L_pa: float       #: pulmonary artery inertance               L_3
    L_pul: float      #: pulmonary inertance                      L_4

    # --- Resistances (mmHg · s · cm^-3) ---------------------------------
    R_aortic_valve: float   #: aortic valve resistance            R_1
    R_ao_sys: float         #: aortic-systemic resistance         R_2
    R_sys: float            #: systemic peripheral resistance     R_3
    R_tricuspid: float      #: tricuspid valve resistance         R_4
    R_pulmonary_valve: float  #: pulmonary valve resistance       R_5
    R_pa: float             #: pulmonary artery resistance        R_6
    R_pul: float            #: pulmonary peripheral resistance    R_7
    R_mitral: float         #: mitral valve resistance            R_8
    R_myo_left: float       #: left myocardial viscosity          R_L
    R_myo_right: float      #: right myocardial viscosity         R_R

    # --- Ventricular elastances (mmHg · cm^-3) --------------------------
    E_d_left: float   #: left diastolic elastance
    E_s_left: float   #: left systolic elastance
    E_d_right: float  #: right diastolic elastance
    E_s_right: float  #: right systolic elastance

    # --- Isovolumetric pressure sources (mmHg) --------------------------
    P0_left: float    #: left peak isovolumetric pressure
    P0_right: float   #: right peak isovolumetric pressure

    # --- Timing and miscellaneous ---------------------------------------
    cardiac_period: float   #: t_c, cardiac period (s)
    aortic_reflux: float    #: regurgitation coefficient in [0, 1)

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #
    def modified(self, **changes) -> "Parameters":
        """Return a new :class:`Parameters` with selected fields changed.

        This is the idiomatic way to set up a parameter-variation study
        (e.g. ``p.modified(R_sys=2.0*p.R_sys)``) without mutating the
        original instance.
        """
        return replace(self, **changes)

    def as_dict(self) -> dict:
        """Shallow dict view, useful for logging or LaTeX table generation."""
        return asdict(self)

    @property
    def systolic_duration(self) -> float:
        """Systolic duration t_s = 0.16 + 0.3 t_c (s)."""
        return 0.16 + 0.3 * self.cardiac_period


# ---------------------------------------------------------------------- #
# Baseline set — matches Table of parameters in the manuscript.          #
# ---------------------------------------------------------------------- #
def baseline_parameters() -> Parameters:
    """Physiological baseline used throughout the manuscript."""
    return Parameters(
        # compliances
        C_ao=0.220,
        C_sys_art=1.460,
        C_sys_ven=20.000,
        C_pa=9.000e-02,
        C_pul_art=2.670,
        C_pul_ven=46.700,
        # inertances
        L_ao_sys=8.250e-04,
        L_sys=3.600e-03,
        L_pa=7.500e-04,
        L_pul=3.080e-03,
        # resistances
        R_aortic_valve=3.7511e-03,
        R_ao_sys=6.7501e-02,
        R_sys=1.000,
        R_tricuspid=3.7511e-03,
        R_pulmonary_valve=3.7511e-03,
        R_pa=3.3761e-02,
        R_pul=0.10130,
        R_mitral=3.751e-03,
        R_myo_left=8.000e-02,
        R_myo_right=1.750e-02,
        # elastances
        E_d_left=0.100,
        E_s_left=1.375,
        E_d_right=3.000e-02,
        E_s_right=0.328,
        # sources
        P0_left=50.0,
        P0_right=24.0,
        # timing / misc
        cardiac_period=0.8,
        aortic_reflux=0.0,
    )

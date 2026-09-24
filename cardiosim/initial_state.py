"""
Initial state of the cardiovascular model.

The state vector x(t) follows the ordering given in Table 1 of the
manuscript:

    x_1  : aortic pressure                    (mmHg)
    x_2  : arterial systemic blood flow       (cm^3 / s)
    x_3  : systemic pressure                  (mmHg)
    x_4  : venous systemic blood flow         (cm^3 / s)
    x_5  : right atrial pressure              (mmHg)
    x_6  : right ventricular volume           (cm^3)
    x_7  : pulmonary venous pressure          (mmHg)
    x_8  : arterial pulmonary blood flow      (cm^3 / s)
    x_9  : pulmonary pressure                 (mmHg)
    x_10 : venous pulmonary blood flow        (cm^3 / s)
    x_11 : left atrial pressure               (mmHg)
    x_12 : left ventricular volume            (cm^3)
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import numpy as np


@dataclass(frozen=True)
class InitialState:
    """Frozen initial-value vector in natural units."""

    P_aortic: float          #: x_1
    Q_sys_art: float         #: x_2
    P_sys: float             #: x_3
    Q_sys_ven: float         #: x_4
    P_right_atrial: float    #: x_5
    V_right_vent: float      #: x_6
    P_pul_venous: float      #: x_7
    Q_pul_art: float         #: x_8
    P_pul: float             #: x_9
    Q_pul_ven: float         #: x_10
    P_left_atrial: float     #: x_11
    V_left_vent: float       #: x_12

    # ------------------------------------------------------------------ #
    def to_array(self) -> np.ndarray:
        """Return the state vector as a length-12 ``numpy.ndarray``."""
        return np.array(
            [
                self.P_aortic, self.Q_sys_art, self.P_sys, self.Q_sys_ven,
                self.P_right_atrial, self.V_right_vent,
                self.P_pul_venous, self.Q_pul_art, self.P_pul, self.Q_pul_ven,
                self.P_left_atrial, self.V_left_vent,
            ],
            dtype=float,
        )

    def modified(self, **changes) -> "InitialState":
        """Return a new :class:`InitialState` with selected fields changed."""
        return replace(self, **changes)

    def as_dict(self) -> dict:
        return asdict(self)


def baseline_initial_state() -> InitialState:
    """Initial values matching the manuscript baseline run."""
    return InitialState(
        P_aortic=71.112,
        Q_sys_art=8.880,
        P_sys=70.516,
        Q_sys_ven=67.337,
        P_right_atrial=3.328,
        V_right_vent=105.520,
        P_pul_venous=13.417,
        Q_pul_art=0.786,
        P_pul=13.393,
        Q_pul_ven=23.836,
        P_left_atrial=11.294,
        V_left_vent=112.760,
    )

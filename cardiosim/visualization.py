"""
Pure plotting functions for cardiosim results.

All functions here take a :class:`~cardiosim.simulate.SimulationResult`
(and optional style kwargs), build a ``matplotlib.figure.Figure``, and
return it. They never save to disk automatically and never call
``plt.show``; the caller decides what to do with the returned figure.

This module replaces the ``Plotter`` class of the earlier codebase.
The design is deliberately functional rather than object-oriented:
plotting responsibilities are *stateless* operations on trajectory
data, and the physiological semantics are already carried by
SimulationResult's named properties.
"""

from __future__ import annotations
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from cardiosim.simulate import SimulationResult
from cardiosim.postprocess import (
    derive_ventricular_pressures,
    last_cycle_slice,
)
from cardiosim.style import COLORS


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _time_window(result: SimulationResult,
                 window: Optional[tuple] = None) -> slice:
    """Return a ``slice`` for a time window (t_start, t_end) in seconds."""
    if window is None:
        return slice(None)
    t = result.t
    i0 = int(np.searchsorted(t, window[0], side="left"))
    i1 = int(np.searchsorted(t, window[1], side="right"))
    return slice(i0, i1)


def _format_time_axis(ax: Axes, label: str = "Time [s]") -> None:
    ax.set_xlabel(label)


# ---------------------------------------------------------------------- #
# Pressure and flow panels                                               #
# ---------------------------------------------------------------------- #
def plot_baseline_overview(
    result: SimulationResult,
    window: Optional[tuple] = (0.0, 1.8),
) -> Figure:
    """Compact 2×2 overview: pressures, flows, volumes, and PV loops.

    This is the figure that should appear in the paper's main text as
    *the* baseline result. It encodes the four canonical views of the
    model in a single panel, with the group's house style.
    """
    sl = _time_window(result, window)
    t = result.t[sl]

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))

    # --- (a) pressures -------------------------------------------------
    ax = axes[0, 0]
    ax.plot(t, result.P_aortic[sl], label="Aortic",
            color=COLORS["navy"], linewidth=1.9)
    ax.plot(t, result.P_sys[sl], label="Arterial systemic",
            color=COLORS["gold"], linewidth=1.6)
    ax.plot(t, result.P_pul[sl], label="Pulmonary",
            color=COLORS["teal"], linewidth=1.6)
    ax.plot(t, result.P_pul_venous[sl], label="Pulmonary venous",
            color=COLORS["maroon"], linewidth=1.4, alpha=0.9)
    ax.set_ylabel("Pressure [mmHg]")
    _format_time_axis(ax)
    ax.set_title("(a) Systemic and pulmonary pressures", loc="left")
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)

    # --- (b) flows -----------------------------------------------------
    ax = axes[0, 1]
    ax.plot(t, result.Q_sys_art[sl], label="Arterial systemic",
            color=COLORS["navy"], linewidth=1.8)
    ax.plot(t, result.Q_pul_art[sl], label="Arterial pulmonary",
            color=COLORS["gold"], linewidth=1.6)
    ax.plot(t, result.Q_sys_ven[sl], label="Venous systemic",
            color=COLORS["teal"], linewidth=1.4)
    ax.plot(t, result.Q_pul_ven[sl], label="Venous pulmonary",
            color=COLORS["maroon"], linewidth=1.4, alpha=0.9)
    ax.set_ylabel("Flow [cm³/s]")
    _format_time_axis(ax)
    ax.set_title("(b) Blood flows", loc="left")
    ax.legend(loc="upper right", ncol=2, columnspacing=1.0)

    # --- (c) ventricular volumes --------------------------------------
    ax = axes[1, 0]
    ax.plot(t, result.V_left_vent[sl], label="Left ventricle",
            color=COLORS["navy"], linewidth=1.9)
    ax.plot(t, result.V_right_vent[sl], label="Right ventricle",
            color=COLORS["gold"], linewidth=1.9)
    ax.set_ylabel("Volume [cm³]")
    _format_time_axis(ax)
    ax.set_title("(c) Ventricular volumes", loc="left")
    ax.legend(loc="lower right")

    # --- (d) PV loops --------------------------------------------------
    ax = axes[1, 1]
    vp = derive_ventricular_pressures(result)
    sl_last = last_cycle_slice(result, n_cycles=1)
    ax.plot(result.V_left_vent[sl_last], vp.P_LV[sl_last],
            label="Left ventricle",
            color=COLORS["navy"], linewidth=2.0)
    ax.plot(result.V_right_vent[sl_last], vp.P_RV[sl_last],
            label="Right ventricle",
            color=COLORS["gold"], linewidth=2.0)
    ax.set_xlabel("Volume [cm³]")
    ax.set_ylabel("Pressure [mmHg]")
    ax.set_title("(d) Pressure–volume loops", loc="left")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_pressures(
    result: SimulationResult,
    window: Optional[tuple] = (0.0, 1.8),
) -> Figure:
    """Single-panel pressure plot (all compartments superposed)."""
    sl = _time_window(result, window)
    t = result.t[sl]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t, result.P_aortic[sl], label="Aortic", color=COLORS["navy"])
    ax.plot(t, result.P_sys[sl], label="Arterial systemic",
            color=COLORS["gold"])
    ax.plot(t, result.P_pul[sl], label="Pulmonary", color=COLORS["teal"])
    ax.plot(t, result.P_pul_venous[sl], label="Pulmonary venous",
            color=COLORS["maroon"])
    ax.plot(t, result.P_right_atrial[sl], label="Right atrial",
            color=COLORS["olive"])
    ax.plot(t, result.P_left_atrial[sl], label="Left atrial",
            color=COLORS["mist"])
    ax.set_ylabel("Pressure [mmHg]")
    _format_time_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def plot_flows(
    result: SimulationResult,
    window: Optional[tuple] = (0.0, 1.8),
) -> Figure:
    """Single-panel flow plot (all compartments superposed)."""
    sl = _time_window(result, window)
    t = result.t[sl]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(t, result.Q_sys_art[sl], label="Arterial systemic",
            color=COLORS["navy"])
    ax.plot(t, result.Q_pul_art[sl], label="Arterial pulmonary",
            color=COLORS["gold"])
    ax.plot(t, result.Q_sys_ven[sl], label="Venous systemic",
            color=COLORS["teal"])
    ax.plot(t, result.Q_pul_ven[sl], label="Venous pulmonary",
            color=COLORS["maroon"])
    ax.set_ylabel("Flow [cm³/s]")
    _format_time_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def plot_pv_loops(
    result: SimulationResult,
    n_cycles: int = 1,
) -> Figure:
    """Pressure–volume loops of both ventricles on the last ``n_cycles``."""
    vp = derive_ventricular_pressures(result)
    sl = last_cycle_slice(result, n_cycles=n_cycles)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(result.V_left_vent[sl], vp.P_LV[sl],
            label="Left ventricle",
            color=COLORS["navy"], linewidth=2.1)
    ax.plot(result.V_right_vent[sl], vp.P_RV[sl],
            label="Right ventricle",
            color=COLORS["gold"], linewidth=2.1)
    ax.set_xlabel("Volume [cm³]")
    ax.set_ylabel("Pressure [mmHg]")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- #
# Parameter variation: baseline + perturbed on the same axes              #
# ---------------------------------------------------------------------- #
def plot_parameter_variation(
    baseline: SimulationResult,
    perturbed: SimulationResult,
    *,
    window: Optional[tuple] = None,
    title: str = "",
    perturbed_label: str = "Perturbed",
) -> Figure:
    """Compare baseline and perturbed runs in a 1×3 panel.

    The baseline trajectory is shown in soft gray behind the perturbed
    one (navy / gold), which is the standard Muro Alto++ pattern for
    didactic parameter-variation figures: the eye anchors on the
    reference and sees the perturbation as a *displacement*.
    """
    if window is None:
        # Default: show the last two cycles of the perturbed run
        t_c = perturbed.parameters.cardiac_period
        t_end = perturbed.t[-1]
        window = (t_end - 2 * t_c, t_end)

    sl_b = _time_window(baseline, window)
    sl_p = _time_window(perturbed, window)
    tb = baseline.t[sl_b]
    tp = perturbed.t[sl_p]

    vp_b = derive_ventricular_pressures(baseline)
    vp_p = derive_ventricular_pressures(perturbed)

    sl_b_last = last_cycle_slice(baseline, n_cycles=1)
    sl_p_last = last_cycle_slice(perturbed, n_cycles=1)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))

    # (a) aortic pressure
    ax = axes[0]
    ax.plot(tb, baseline.P_aortic[sl_b], color=COLORS["mist"],
            linewidth=1.3, label="Baseline")
    ax.plot(tp, perturbed.P_aortic[sl_p], color=COLORS["navy"],
            linewidth=1.9, label=perturbed_label)
    ax.set_ylabel("Aortic pressure [mmHg]")
    _format_time_axis(ax)
    ax.set_title("(a) Aortic pressure", loc="left")
    ax.legend(loc="best")

    # (b) arterial systemic flow
    ax = axes[1]
    ax.plot(tb, baseline.Q_sys_art[sl_b], color=COLORS["mist"],
            linewidth=1.3, label="Baseline")
    ax.plot(tp, perturbed.Q_sys_art[sl_p], color=COLORS["gold"],
            linewidth=1.9, label=perturbed_label)
    ax.set_ylabel("Arterial systemic flow [cm³/s]")
    _format_time_axis(ax)
    ax.set_title("(b) Arterial systemic flow", loc="left")
    ax.legend(loc="best")

    # (c) LV pressure–volume loop
    ax = axes[2]
    ax.plot(baseline.V_left_vent[sl_b_last], vp_b.P_LV[sl_b_last],
            color=COLORS["mist"], linewidth=1.6,
            label="Baseline")
    ax.plot(perturbed.V_left_vent[sl_p_last], vp_p.P_LV[sl_p_last],
            color=COLORS["navy"], linewidth=2.1,
            label=perturbed_label)
    ax.set_xlabel("Volume [cm³]")
    ax.set_ylabel("LV pressure [mmHg]")
    ax.set_title("(c) Left-ventricular PV loop", loc="left")
    ax.legend(loc="upper right")

    if title:
        fig.suptitle(title, fontsize=11.5, color=COLORS["navy"], y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- #
# Activation visualizer (useful didactic figure)                         #
# ---------------------------------------------------------------------- #
def plot_activation_function(
    result: SimulationResult,
    n_cycles: int = 2,
) -> Figure:
    """Plot a(t), E_L(t), E_R(t) to show what the ventricles 'feel'."""
    t_c = result.parameters.cardiac_period
    t = np.linspace(0, n_cycles * t_c, 2000)
    m = result.model
    a = m.activation(t)
    E_L = m.elastance_left(t)
    E_R = m.elastance_right(t)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(t, a, color=COLORS["navy"], label="a(t)", linewidth=2.0)
    ax.set_ylabel("Activation a(t)", color=COLORS["navy"])
    ax.set_xlabel("Time [s]")
    ax.tick_params(axis="y", labelcolor=COLORS["navy"])

    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(t, E_L, color=COLORS["gold"], linewidth=1.8,
             label="E_L(t)")
    ax2.plot(t, E_R, color=COLORS["maroon"], linewidth=1.8,
             linestyle="--", label="E_R(t)")
    ax2.set_ylabel("Elastance [mmHg/cm³]", color=COLORS["slate"])

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- #
# Multi-panel figures matching the manuscript layout                      #
# ---------------------------------------------------------------------- #
def plot_pressures_multipanel(
    result: SimulationResult,
    window: Optional[tuple] = (0.0, 1.8),
) -> Figure:
    """2×3 panel of individual pressure signals (replaces manuscript Fig. 6).

    Each sub-plot shares the same time axis but has an independent
    pressure scale, which makes it possible to read the dynamical
    signature of each compartment without the smallest signals being
    swallowed by the aortic pulse.
    """
    sl = _time_window(result, window)
    t = result.t[sl]

    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.4), sharex=True)

    entries = [
        ("Aortic",             result.P_aortic,         COLORS["navy"]),
        ("Arterial systemic",  result.P_sys,            COLORS["gold"]),
        ("Pulmonary",          result.P_pul,            COLORS["teal"]),
        ("Pulmonary venous",   result.P_pul_venous,     COLORS["maroon"]),
        ("Right atrial",       result.P_right_atrial,   COLORS["olive"]),
        ("Left atrial",        result.P_left_atrial,    COLORS["mist"]),
    ]

    labels = "abcdef"
    for ax, (name, data, color), lab in zip(axes.flat, entries, labels):
        ax.plot(t, data[sl], color=color, linewidth=1.8)
        ax.set_title(f"({lab}) {name}", loc="left", fontsize=10.5)
        ax.set_ylabel("Pressure [mmHg]")

    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    return fig


def plot_ventricular_volumes(
    result: SimulationResult,
    window: Optional[tuple] = (0.0, 1.8),
) -> Figure:
    """Side-by-side ventricular volumes (replaces manuscript Fig. 8)."""
    sl = _time_window(result, window)
    t = result.t[sl]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)

    axes[0].plot(t, result.V_left_vent[sl],
                 color=COLORS["navy"], linewidth=2.0)
    axes[0].set_title("(a) Left ventricle", loc="left")
    axes[0].set_ylabel("Volume [cm³]")
    axes[0].set_xlabel("Time [s]")

    axes[1].plot(t, result.V_right_vent[sl],
                 color=COLORS["gold"], linewidth=2.0)
    axes[1].set_title("(b) Right ventricle", loc="left")
    axes[1].set_xlabel("Time [s]")

    fig.tight_layout()
    return fig


def plot_flows_multipanel(
    result: SimulationResult,
    window: Optional[tuple] = (0.0, 1.8),
) -> Figure:
    """2×2 panel of individual flow signals (replaces manuscript Fig. 9)."""
    sl = _time_window(result, window)
    t = result.t[sl]

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.4), sharex=True)

    entries = [
        ("Venous systemic",      result.Q_sys_ven,  COLORS["teal"]),
        ("Venous pulmonary",     result.Q_pul_ven,  COLORS["maroon"]),
        ("Arterial systemic",    result.Q_sys_art,  COLORS["navy"]),
        ("Arterial pulmonary",   result.Q_pul_art,  COLORS["gold"]),
    ]

    labels = "abcd"
    for ax, (name, data, color), lab in zip(axes.flat, entries, labels):
        ax.plot(t, data[sl], color=color, linewidth=1.8)
        ax.set_title(f"({lab}) {name}", loc="left", fontsize=10.5)
        ax.set_ylabel("Flow [cm³/s]")

    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    return fig

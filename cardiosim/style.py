"""
Visual identity for cardiosim figures.

Colour palette follows the Nonlinear Dynamics and Signal Processing
Group identity:

    navy      = '#0A2240'   primary text, main signal
    gold      = '#C8A951'   accent, highlights, comparison curves
    softgray  = '#EDF2F7'   inset / zoom backgrounds
    teal      = '#2E5266'   secondary signal
    maroon    = '#7A1F2B'   tertiary signal / tensor-stress cases
    olive     = '#5C5A3A'   fourth-order / extra
    slate     = '#3B3F45'   axis / annotation grey

The palette is chosen to be (a) print-safe, (b) colour-vision friendly
for the most common deficiencies, and (c) visually distinct from the
Matplotlib default ``tab10``.

Usage
-----
    from cardiosim.style import apply_house_style, COLORS
    apply_house_style()
"""

from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt


COLORS = {
    "navy":     "#0A2240",
    "gold":     "#C8A951",
    "softgray": "#EDF2F7",
    "teal":     "#2E5266",
    "maroon":   "#7A1F2B",
    "olive":    "#5C5A3A",
    "slate":    "#3B3F45",
    "mist":     "#94A3B8",
}

# Ordered cycle for plots with up to six simultaneous signals
CYCLE = [
    COLORS["navy"],
    COLORS["gold"],
    COLORS["teal"],
    COLORS["maroon"],
    COLORS["olive"],
    COLORS["mist"],
]


def apply_house_style() -> None:
    """Apply the group's visual identity to all subsequent figures."""
    mpl.rcParams.update({
        # Typography
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "IBM Plex Sans", "DejaVu Sans", "Arial"],
        "font.size":        10.0,
        "axes.titlesize":   11.0,
        "axes.labelsize":   10.0,
        "xtick.labelsize":   9.0,
        "ytick.labelsize":   9.0,
        "legend.fontsize":   9.0,
        "figure.titlesize": 12.0,

        # Axes and grid
        "axes.edgecolor":     COLORS["slate"],
        "axes.labelcolor":    COLORS["slate"],
        "axes.titlecolor":    COLORS["navy"],
        "axes.titleweight":   "semibold",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.9,
        "axes.grid":          True,
        "grid.linestyle":     ":",
        "grid.linewidth":     0.6,
        "grid.color":         COLORS["mist"],
        "grid.alpha":         0.55,

        # Ticks
        "xtick.color":        COLORS["slate"],
        "ytick.color":        COLORS["slate"],
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,

        # Lines
        "lines.linewidth":    1.7,
        "lines.solid_capstyle": "round",

        # Legends
        "legend.frameon":     False,
        "legend.borderpad":   0.2,
        "legend.handlelength": 1.6,

        # Figures
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "figure.facecolor":   "white",

        # Colour cycle
        "axes.prop_cycle":    mpl.cycler(color=CYCLE),
    })

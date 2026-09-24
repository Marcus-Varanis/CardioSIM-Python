# cardiosim

**A teaching-oriented closed-loop cardiovascular model in Python.**

Companion code to the manuscript

> *A Closed-Loop Cardiovascular Model as a Teaching Framework for Physics,
> Mathematical Modeling, and Scientific Computing.*
> Revista Brasileira de Ensino de Física.

This package reimplements, refactors, and documents the 12-state
lumped-parameter cardiovascular model described in the paper, as a
reproducible computational framework suitable for classroom use.

## Design principles

1. **Immutable inputs.** `Parameters` and `InitialState` are frozen
   dataclasses, so the numerical provenance of every simulation is
   fully traceable.
2. **Separation of concerns.** Model definition (`model.py`), numerical
   integration (`simulate.py`), post-processing (`postprocess.py`), and
   visualization (`visualization.py`) live in disjoint modules.
3. **Single source of truth for derived quantities.** Ventricular
   pressures `P_LV(t)`, `P_RV(t)` are derived from the simulation
   result in exactly one place (`derive_ventricular_pressures`). This
   removes the silent discrepancy present in earlier implementations
   that plotted only the active source `P_0 a(t)`.
4. **House visual identity.** All figures use a fixed, print-safe
   colour palette (navy `#0A2240`, gold `#C8A951`, soft gray
   `#EDF2F7`) and a consistent typographic treatment.

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`.

## Quick start

```python
from cardiosim import (
    baseline_parameters, baseline_initial_state,
    run_simulation, cycle_metrics,
)
from cardiosim.style import apply_house_style
from cardiosim.visualization import plot_baseline_overview

apply_house_style()

p  = baseline_parameters()
x0 = baseline_initial_state()

result = run_simulation(p, x0, t_end=10.0)

m = cycle_metrics(result)
print(f"MAP = {m.MAP:.1f} mmHg, EF = {m.EF_LV:.1f}%, CO = {m.CO:.2f} L/min")

fig = plot_baseline_overview(result)
fig.savefig("overview.pdf")
```

## Reproducing the manuscript figures

```bash
python examples/01_baseline.py               # baseline figures
python examples/02_parameter_variation.py    # Figs. 12–14
python examples/03_solver_benchmark.py       # Appendix-B benchmark
python examples/04_reproduce_paper_figures.py  # all paper figures
```

Output is written under `figures/` or `paper_figures/`.

## A parameter-variation study in three lines

```python
p_hyper = p.modified(R_sys=2.0 * p.R_sys)
r_hyper = run_simulation(p_hyper, x0, t_end=15.0)
plot_parameter_variation(result, r_hyper,
                         perturbed_label="R_sys × 2.0").savefig(...)
```

## Project layout

```
cardiosim/
├── cardiosim/           ← the package
│   ├── parameters.py
│   ├── initial_state.py
│   ├── model.py
│   ├── simulate.py
│   ├── postprocess.py
│   ├── visualization.py
│   └── style.py
├── examples/
│   ├── 01_baseline.py
│   ├── 02_parameter_variation.py
│   ├── 03_solver_benchmark.py
│   └── 04_reproduce_paper_figures.py
├── tests/
│   └── test_sanity.py
└── pyproject.toml
```

## Tests

```bash
pytest -q
```

The suite checks: RHS shape and finiteness, completion of a 10-s run,
periodicity of the solution after transient dies out, and baseline
hemodynamic metrics against the reference ranges reported in Table 2
of the manuscript.

## Citation

If you use this code, please cite the accompanying RBEF paper and the
archived release (DOI minted upon publication via Zenodo).

## License

MIT.

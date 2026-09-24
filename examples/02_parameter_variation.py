"""
Example 02 — Parameter-variation studies.

Reproduces the three perturbations discussed in the manuscript
(reduced / increased systemic resistance, aortic regurgitation) in
the Muro Alto++ "baseline vs. perturbed" format: the baseline
trajectory is rendered in soft gray and the perturbed one is overlaid
in navy / gold.
"""
from pathlib import Path

from cardiosim import (
    baseline_parameters,
    baseline_initial_state,
    run_simulation,
)
from cardiosim.style import apply_house_style
from cardiosim.visualization import plot_parameter_variation


def main():
    apply_house_style()

    p = baseline_parameters()
    x0 = baseline_initial_state()

    print("Running baseline ...")
    baseline = run_simulation(p, x0, t_end=15.0)

    # --- Study 1: reduced systemic resistance (-50%) ------------------
    print("Study 1: R_sys reduced by 50%")
    p1 = p.modified(R_sys=0.5 * p.R_sys)
    r1 = run_simulation(p1, x0, t_end=15.0)

    # --- Study 2: increased systemic resistance (+100%) ---------------
    print("Study 2: R_sys doubled")
    p2 = p.modified(R_sys=2.0 * p.R_sys)
    r2 = run_simulation(p2, x0, t_end=15.0)

    # --- Study 3: aortic regurgitation --------------------------------
    print("Study 3: aortic regurgitation (Ref=0.05, R_ao_valve x0.4)")
    p3 = p.modified(
        aortic_reflux=0.05,
        R_aortic_valve=0.4 * p.R_aortic_valve,
    )
    r3 = run_simulation(p3, x0, t_end=15.0)

    out = Path("figures")
    out.mkdir(exist_ok=True)

    fig = plot_parameter_variation(
        baseline, r1,
        title="Reduced systemic resistance (R_sys × 0.5)",
        perturbed_label="R_sys × 0.5",
    )
    fig.savefig(out / "var_R_sys_reduced.png")
    fig.savefig(out / "var_R_sys_reduced.pdf")

    fig = plot_parameter_variation(
        baseline, r2,
        title="Increased systemic resistance (R_sys × 2.0)",
        perturbed_label="R_sys × 2.0",
    )
    fig.savefig(out / "var_R_sys_increased.png")
    fig.savefig(out / "var_R_sys_increased.pdf")

    fig = plot_parameter_variation(
        baseline, r3,
        title="Modified aortic valve (regurgitation)",
        perturbed_label="Regurgitant",
    )
    fig.savefig(out / "var_aortic_valve.png")
    fig.savefig(out / "var_aortic_valve.pdf")

    print(f"\nParameter-variation figures saved under {out.resolve()}")


if __name__ == "__main__":
    main()

"""
Example 04 — Reproduce the paper's figures (Muro Alto++ edition).

Regenerates every figure referenced in the main text of the RBEF
manuscript using the refactored pipeline. Output filenames match the
figure numbers of the manuscript for one-to-one replacement:

    fig06_baseline_pressures.{png,pdf}   — manuscript Fig. 6
    fig07_pressures_superposed.{png,pdf} — manuscript Fig. 7
    fig08_ventricular_volumes.{png,pdf}  — manuscript Fig. 8
    fig09_baseline_flows.{png,pdf}       — manuscript Fig. 9
    fig10_flows_superposed.{png,pdf}     — manuscript Fig. 10
    fig11_pv_loops.{png,pdf}             — manuscript Fig. 11 (CORRECTED)
    fig12_var_R_sys_reduced.{png,pdf}    — manuscript Fig. 12
    fig13_var_R_sys_increased.{png,pdf}  — manuscript Fig. 13
    fig14_var_aortic_valve.{png,pdf}     — manuscript Fig. 14

Additional figures (not in the original manuscript but recommended
for the revised submission):

    figEX_baseline_overview.{png,pdf}    — 2x2 summary panel
    figEX_activation.{png,pdf}           — a(t), E_L(t), E_R(t)
"""
from pathlib import Path

from cardiosim import (
    baseline_parameters,
    baseline_initial_state,
    run_simulation,
    cycle_metrics,
)
from cardiosim.style import apply_house_style
from cardiosim.visualization import (
    plot_baseline_overview,
    plot_pressures,
    plot_pressures_multipanel,
    plot_flows,
    plot_flows_multipanel,
    plot_ventricular_volumes,
    plot_pv_loops,
    plot_activation_function,
    plot_parameter_variation,
)


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")


def main():
    apply_house_style()

    p = baseline_parameters()
    x0 = baseline_initial_state()

    print("Running baseline (t_end = 15 s) ...")
    baseline = run_simulation(p, x0, t_end=15.0)

    m = cycle_metrics(baseline)
    print(f"  MAP = {m.MAP:.2f} mmHg, SV = {m.SV_LV:.2f} mL, "
          f"EF = {m.EF_LV:.2f} %, CO = {m.CO:.2f} L/min")

    # Parameter-variation runs
    print("Running parameter-variation studies ...")
    r_low  = run_simulation(p.modified(R_sys=0.5 * p.R_sys),
                            x0, t_end=15.0)
    r_high = run_simulation(p.modified(R_sys=2.0 * p.R_sys),
                            x0, t_end=15.0)
    r_regurg = run_simulation(
        p.modified(aortic_reflux=0.05,
                   R_aortic_valve=0.4 * p.R_aortic_valve),
        x0, t_end=15.0,
    )

    out = Path("paper_figures")
    out.mkdir(exist_ok=True)

    # --- Figures matching manuscript numbering -----------------------
    print("Generating paper figures ...")
    save(plot_pressures_multipanel(baseline),        out, "fig06_baseline_pressures")
    save(plot_pressures(baseline),                   out, "fig07_pressures_superposed")
    save(plot_ventricular_volumes(baseline),         out, "fig08_ventricular_volumes")
    save(plot_flows_multipanel(baseline),            out, "fig09_baseline_flows")
    save(plot_flows(baseline),                       out, "fig10_flows_superposed")
    save(plot_pv_loops(baseline),                    out, "fig11_pv_loops")

    save(
        plot_parameter_variation(
            baseline, r_low,
            title="Reduced systemic resistance (R_sys × 0.5)",
            perturbed_label="R_sys × 0.5",
        ),
        out, "fig12_var_R_sys_reduced",
    )

    save(
        plot_parameter_variation(
            baseline, r_high,
            title="Increased systemic resistance (R_sys × 2.0)",
            perturbed_label="R_sys × 2.0",
        ),
        out, "fig13_var_R_sys_increased",
    )

    save(
        plot_parameter_variation(
            baseline, r_regurg,
            title="Modified aortic valve (regurgitation)",
            perturbed_label="Regurgitant",
        ),
        out, "fig14_var_aortic_valve",
    )

    # --- Extra figures recommended for the revised submission --------
    save(plot_baseline_overview(baseline),  out, "figEX_baseline_overview")
    save(plot_activation_function(baseline), out, "figEX_activation")

    print(f"\nAll paper figures saved under {out.resolve()}")


if __name__ == "__main__":
    main()

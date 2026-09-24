"""
Example 01 — Baseline simulation.

Reproduces the baseline figures of the manuscript using the refactored
pipeline:

    parameters + initial state  ->  run_simulation  ->  plotting

All figures are saved to ./figures/ with the group's visual identity.
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
    plot_flows,
    plot_pv_loops,
    plot_activation_function,
)


def main():
    apply_house_style()

    parameters = baseline_parameters()
    initial_state = baseline_initial_state()

    # A 10 s simulation is ample: ~12.5 cardiac cycles at t_c = 0.8 s.
    result = run_simulation(
        parameters,
        initial_state,
        t_end=10.0,
        solver="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    # Print hemodynamic summary on the last stable cycle
    m = cycle_metrics(result)
    print("---- Baseline hemodynamic metrics (last cycle) ----")
    print(f"  Systolic aortic pressure : {m.P_syst_aortic:6.2f} mmHg")
    print(f"  Diastolic aortic pressure: {m.P_diast_aortic:6.2f} mmHg")
    print(f"  Pulse pressure           : {m.pulse_pressure:6.2f} mmHg")
    print(f"  Mean arterial pressure   : {m.MAP:6.2f} mmHg")
    print(f"  Stroke volume  (LV)      : {m.SV_LV:6.2f} mL")
    print(f"  Ejection fraction (LV)   : {m.EF_LV:6.2f} %")
    print(f"  Heart rate               : {m.HR:6.2f} bpm")
    print(f"  Cardiac output           : {m.CO:6.2f} L/min")
    print(f"  Solver wall time         : {result.wall_time_s:6.3f} s")
    print(f"  RHS evaluations          : {result.nfev}")

    out = Path("figures")
    out.mkdir(exist_ok=True)

    fig = plot_baseline_overview(result)
    fig.savefig(out / "baseline_overview.png")
    fig.savefig(out / "baseline_overview.pdf")

    fig = plot_pressures(result)
    fig.savefig(out / "baseline_pressures.png")
    fig.savefig(out / "baseline_pressures.pdf")

    fig = plot_flows(result)
    fig.savefig(out / "baseline_flows.png")
    fig.savefig(out / "baseline_flows.pdf")

    fig = plot_pv_loops(result)
    fig.savefig(out / "baseline_pv_loops.png")
    fig.savefig(out / "baseline_pv_loops.pdf")

    fig = plot_activation_function(result)
    fig.savefig(out / "baseline_activation.png")
    fig.savefig(out / "baseline_activation.pdf")

    print(f"\nFigures saved under {out.resolve()}")


if __name__ == "__main__":
    main()

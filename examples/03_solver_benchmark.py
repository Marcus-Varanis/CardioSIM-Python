"""
Example 03 — Solver benchmark.

Reproduces (and cleans up) the benchmark table of the original
Appendix B. Runs RK45, LSODA, Radau, and DOP853 over the same 10-s
window with matched tolerances, and prints a formatted comparison.
"""
from cardiosim import (
    baseline_parameters,
    baseline_initial_state,
    run_simulation,
)


def main():
    p = baseline_parameters()
    x0 = baseline_initial_state()

    solvers = ["RK45", "DOP853", "LSODA", "Radau", "BDF"]
    rtol = 1e-10
    atol = 1e-12
    t_end = 10.0

    print(f"Benchmark over {t_end:g}-s simulation window "
          f"(rtol={rtol:.0e}, atol={atol:.0e})\n")
    print(f"{'Solver':<8}  {'Steps':>8}  {'nfev':>8}  "
          f"{'Time [s]':>10}  {'Status'}")
    print("-" * 55)

    for name in solvers:
        try:
            res = run_simulation(
                p, x0, t_end=t_end,
                solver=name, rtol=rtol, atol=atol,
            )
            status = "success"
            print(f"{name:<8}  {res.nsteps:>8d}  {res.nfev:>8d}  "
                  f"{res.wall_time_s:>10.3f}  {status}")
        except Exception as exc:
            print(f"{name:<8}  failed: {exc}")


if __name__ == "__main__":
    main()

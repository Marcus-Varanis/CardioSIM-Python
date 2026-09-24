"""
Sanity tests for cardiosim.

These are *not* full clinical validation tests. They check that:
  (i) the 12-state ODE right-hand side has the expected shape;
 (ii) a baseline simulation reaches a periodic regime;
(iii) the hemodynamic metrics fall within the reference physiological
     ranges reported in the manuscript.

Run with:
    pytest -q tests/
"""
import numpy as np
import pytest

from cardiosim import (
    baseline_parameters,
    baseline_initial_state,
    run_simulation,
    cycle_metrics,
    derive_ventricular_pressures,
)


# ---------------------------------------------------------------------- #
# Fixtures                                                               #
# ---------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def baseline_result():
    """Shared baseline simulation (10 s, RK45, tight tolerances)."""
    p = baseline_parameters()
    x0 = baseline_initial_state()
    return run_simulation(p, x0, t_end=10.0, solver="RK45")


# ---------------------------------------------------------------------- #
# Tests                                                                  #
# ---------------------------------------------------------------------- #
def test_rhs_shape():
    """RHS must return a length-12 vector."""
    p = baseline_parameters()
    x0 = baseline_initial_state()
    from cardiosim.model import ClosedLoopCardiovascular
    model = ClosedLoopCardiovascular(p, x0)
    dydt = model.rhs(0.0, x0.to_array())
    assert dydt.shape == (12,)
    assert np.all(np.isfinite(dydt))


def test_simulation_completes(baseline_result):
    assert baseline_result.t[-1] >= 10.0 - 1e-9
    assert baseline_result.y.shape[0] == 12
    assert np.all(np.isfinite(baseline_result.y))


def test_periodic_regime(baseline_result):
    """Compare the penultimate and last full cycles of LV volume.

    After 10 s ≈ 12.5 cardiac cycles at t_c = 0.8 s, the solution
    should be close to periodic. Because the adaptive solver does not
    produce matching time grids across cycles, both cycles are first
    resampled to a common uniform grid before comparison.
    """
    t_c = baseline_result.parameters.cardiac_period
    t = baseline_result.t
    V = baseline_result.V_left_vent
    t_end = t[-1]

    # Common reference time within a cycle (relative coordinate)
    tau = np.linspace(0.0, t_c, 400)

    # Last full cycle: t in [t_end - t_c, t_end]
    mask_last = (t >= t_end - t_c) & (t <= t_end)
    t_last = t[mask_last] - (t_end - t_c)
    V_last = np.interp(tau, t_last, V[mask_last])

    # Penultimate cycle: t in [t_end - 2 t_c, t_end - t_c]
    mask_prev = (t >= t_end - 2 * t_c) & (t <= t_end - t_c)
    t_prev = t[mask_prev] - (t_end - 2 * t_c)
    V_prev = np.interp(tau, t_prev, V[mask_prev])

    err = np.max(np.abs(V_last - V_prev)) / np.ptp(V_last)
    assert err < 5e-3, f"LV volume not yet periodic: relative err = {err:.3e}"


def test_metrics_within_physiological_ranges(baseline_result):
    m = cycle_metrics(baseline_result)
    # Reference ranges (adult at rest) from the manuscript's Table 2
    assert 90.0 < m.P_syst_aortic < 140.0
    assert 55.0 < m.P_diast_aortic < 95.0
    assert 30.0 < m.pulse_pressure < 70.0
    assert 70.0 < m.MAP < 110.0
    assert 40.0 < m.SV_LV < 120.0
    assert 40.0 < m.EF_LV < 80.0
    assert 3.0 < m.CO < 7.5


def test_pv_loop_has_expected_topology(baseline_result):
    """A proper PV loop is *enclosed*: max P is reached at low-to-mid V,
    not at maximum V (which would be the case of the buggy source-only
    plot)."""
    vp = derive_ventricular_pressures(baseline_result)
    from cardiosim.postprocess import last_cycle_slice
    sl = last_cycle_slice(baseline_result)
    V = baseline_result.V_left_vent[sl]
    P = vp.P_LV[sl]
    # Index where pressure peaks
    i_peak = int(np.argmax(P))
    V_at_peak = V[i_peak]
    V_min, V_max = V.min(), V.max()
    # Peak pressure should occur during ejection (V between min and max,
    # typically closer to mid-range — certainly not at V_max).
    assert V_min < V_at_peak < V_max
    assert (V_at_peak - V_min) / (V_max - V_min) < 0.9

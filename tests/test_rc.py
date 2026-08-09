"""Property tests for the 2R2C thermal model (src/cooling_twin/models/rc.py).

A property test checks an invariant across MANY randomly generated
inputs, not one hardcoded example -- test pattern 2 of 4
(05_ENGINEERING_STANDARDS.md SS3). The two properties checked here are
both named in 03_DOMAIN_REFERENCE.md: INV-8 (energy balance) and the
physical fact that a network built only from resistors and capacitors
(no inductance) has real, negative eigenvalues and therefore cannot
ring -- its step response must move monotonically toward steady state.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from cooling_twin import set_seed
from cooling_twin.models.rc import RCParams, simulate

N_PROPERTY_DRAWS = 50
HORIZON_HOURS = 48.0
ENERGY_RELATIVE_TOLERANCE = 0.01  # 1%, matches L4.1's INV-8 check

# NOT zero: solve_ivp's RK45 + t_eval dense-output interpolation can
# introduce a small spurious dip near a flat asymptote even though the
# true (eigen-decomposition) solution is provably monotonic for this
# initial condition -- verified independently against the analytic
# solution while writing this test. Empirically, the worst case across
# 300 draws at hourly resolution in this parameter range was ~0.20 degC;
# this tolerance is a solver-resolution allowance, not permission for
# the model to be wrong. See "why it's written this way" in the L4.5
# lesson for the investigation that produced this number.
MONOTONICITY_TOLERANCE_C = 0.5


# --- Unit tests: input validation (test pattern 1 of 4, L3.2's pattern) ---
# The property tests below only ever construct VALID RCParams/inputs by
# design -- these are the tests that exercise the raise branches.


@pytest.mark.parametrize("field", ["r_ie", "r_ea", "c_i", "c_e"])
def test_rcparams_rejects_non_positive_field(field: str) -> None:
    """INV-5: every field must be strictly positive."""
    valid = {"r_ie": 2e-4, "r_ea": 5e-4, "c_i": 5e6, "c_e": 3e7}
    valid[field] = 0.0
    with pytest.raises(ValueError, match="INV-5"):
        RCParams(**valid)


def test_simulate_rejects_mismatched_lengths() -> None:
    params = RCParams(r_ie=2e-4, r_ea=5e-4, c_i=5e6, c_e=3e7)
    t_hours = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="same length"):
        simulate(params, t_hours, np.zeros(2), np.zeros(3), np.array([30.0, 30.0]))


def test_simulate_rejects_fewer_than_two_points() -> None:
    params = RCParams(r_ie=2e-4, r_ea=5e-4, c_i=5e6, c_e=3e7)
    t_hours = np.array([0.0])
    with pytest.raises(ValueError, match="at least 2 points"):
        simulate(params, t_hours, np.zeros(1), np.zeros(1), np.array([30.0, 30.0]))


def test_simulate_rejects_non_increasing_t_hours() -> None:
    params = RCParams(r_ie=2e-4, r_ea=5e-4, c_i=5e6, c_e=3e7)
    t_hours = np.array([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        simulate(params, t_hours, np.zeros(3), np.zeros(3), np.array([30.0, 30.0]))


# --- Property tests (test pattern 2 of 4) ---


def _random_params_and_step(rng: np.random.Generator) -> tuple[RCParams, float, float]:
    """One randomly drawn (RCParams, step forcing) triple.

    Range is physically motivated, not arbitrary: same order of
    magnitude as every R/C used in L4.1-L4.4, and r_ea sampled to
    dominate r_ie (envelope-to-ambient insulation resistance exceeding
    indoor-to-envelope resistance is true for any normally-insulated
    building; fully unconstrained sampling produces networks with no
    real building behind them).
    """
    r_ie = 10 ** rng.uniform(-4, -3.3)
    r_ea = 10 ** rng.uniform(-3.5, -3.0)
    c_i = 10 ** rng.uniform(6, 6.7)
    c_e = 10 ** rng.uniform(7, 7.5)
    params = RCParams(r_ie=r_ie, r_ea=r_ea, c_i=c_i, c_e=c_e)
    q_step = float(rng.choice([-1.0, 1.0]) * rng.uniform(2e4, 6e4))
    t_out_val = float(rng.uniform(20.0, 35.0))
    return params, q_step, t_out_val


def _energy_relative_error(
    params: RCParams,
    t_hours: np.ndarray,
    t_i: np.ndarray,
    t_e: np.ndarray,
    q_step: float,
    t_out_val: float,
) -> float:
    """INV-8, 2-node form: energy in minus energy lost through R_ea must
    equal energy stored in both nodes. R_ie is internal -- heat leaving
    node i through it exactly equals heat entering node e through it --
    so only R_ea (the true system boundary, connecting to T_out)
    contributes a loss term.
    """
    t_seconds = t_hours * 3600.0
    energy_in = q_step * t_seconds[-1]
    energy_lost = np.trapezoid((t_e - t_out_val) / params.r_ea, t_seconds)
    energy_stored = params.c_i * (t_i[-1] - t_i[0]) + params.c_e * (t_e[-1] - t_e[0])
    return float(abs(energy_in - energy_lost - energy_stored) / abs(energy_in))


def test_energy_conservation_property() -> None:
    """For any physically valid params and step forcing, INV-8 holds."""
    rng = set_seed()
    t_hours = np.arange(0.0, HORIZON_HOURS + 0.01, 1.0)

    for _ in range(N_PROPERTY_DRAWS):
        params, q_step, t_out_val = _random_params_and_step(rng)
        q_gain_w = np.full_like(t_hours, q_step)
        t_out_c = np.full_like(t_hours, t_out_val)
        initial_state = np.array([t_out_val, t_out_val])

        sim = simulate(params, t_hours, q_gain_w, t_out_c, initial_state)
        t_i, t_e = sim["t_i"].to_numpy(), sim["t_e"].to_numpy()

        relative_error = _energy_relative_error(params, t_hours, t_i, t_e, q_step, t_out_val)
        assert relative_error < ENERGY_RELATIVE_TOLERANCE, (
            f"INV-8 violated: relative error {relative_error:.4f} for "
            f"{params}, q_step={q_step:.1f}, t_out={t_out_val:.1f}"
        )


def test_step_response_monotonic() -> None:
    """Starting from equilibrium (T_i0=T_e0=T_out), T_i must move
    monotonically toward its steady state for any physically valid
    params and step forcing -- an RC-only network cannot ring.
    """
    rng = set_seed()
    t_hours = np.arange(0.0, HORIZON_HOURS + 0.01, 1.0)

    for _ in range(N_PROPERTY_DRAWS):
        params, q_step, t_out_val = _random_params_and_step(rng)
        q_gain_w = np.full_like(t_hours, q_step)
        t_out_c = np.full_like(t_hours, t_out_val)
        initial_state = np.array([t_out_val, t_out_val])

        sim = simulate(params, t_hours, q_gain_w, t_out_c, initial_state)
        t_i = sim["t_i"].to_numpy()

        direction = 1.0 if q_step > 0 else -1.0
        signed_diffs = direction * np.diff(t_i)
        assert signed_diffs.min() > -MONOTONICITY_TOLERANCE_C, (
            f"Step response not monotonic (min step {signed_diffs.min():.4f} degC) "
            f"for {params}, q_step={q_step:.1f}, t_out={t_out_val:.1f}"
        )


def _broken_rc_derivatives_sign_flip(
    t: float,
    state: np.ndarray,
    params: RCParams,
    q_gain: Callable[[float], float],
    t_out: Callable[[float], float],
) -> np.ndarray:
    """A deliberately WRONG 2R2C: the R_ie coupling term has a `+`
    where `rc_derivatives()` has a `-` in the T_i equation -- the exact
    mistake L4.2's "why it's written this way" section warns about
    (the same term must appear with opposite sign in each node's
    equation, or energy is manufactured at the boundary). Exists only
    to prove the property tests above would catch it; never used
    outside this test.
    """
    t_i, t_e = state
    q, t_amb = q_gain(t), t_out(t)
    d_t_i = (q + (t_i - t_e) / params.r_ie) / params.c_i  # WRONG: should be `-`
    d_t_e = ((t_i - t_e) / params.r_ie - (t_e - t_amb) / params.r_ea) / params.c_e
    return np.array([d_t_i, d_t_e])


def _integrate_with(
    deriv_fn: Callable[..., np.ndarray],
    params: RCParams,
    t_hours: np.ndarray,
    q_gain_w: np.ndarray,
    t_out_c: np.ndarray,
    initial_state: np.ndarray,
) -> np.ndarray:
    """Same solve_ivp wiring as simulate() (L4.3), but accepts any
    derivative function -- simulate() itself is hardcoded to
    rc_derivatives, so driving the deliberately broken model above
    through the identical integration path needs this instead.
    """
    t_seconds = (t_hours - t_hours[0]) * 3600.0

    def q_gain(t: float) -> float:
        return float(np.interp(t, t_seconds, q_gain_w))

    def t_out(t: float) -> float:
        return float(np.interp(t, t_seconds, t_out_c))

    result = solve_ivp(
        deriv_fn,
        (t_seconds[0], t_seconds[-1]),
        initial_state,
        method="RK45",
        t_eval=t_seconds,
        args=(params, q_gain, t_out),
    )
    assert result.success, f"solve_ivp failed on the broken model: {result.message}"
    return result.y


def test_property_tests_catch_a_sign_flip_bug() -> None:
    """Proof the properties above have teeth, not just prose.

    Runs the SAME energy-conservation check against the sign-flipped
    model and confirms it fails -- badly (a positive-feedback loop, not
    a bounded error: T_i runs away toward +-1e40+ degC within 48h).
    Also confirms the sign flip is caught by ENERGY conservation but
    NOT by monotonicity -- the runaway happens to stay in the correct
    direction, it's just unboundedly fast. That asymmetry is exactly
    why L4.5 tests both properties instead of just one; a suite with
    only the monotonicity test would have shipped this bug.
    """
    rng = set_seed()
    params, q_step, t_out_val = _random_params_and_step(rng)
    t_hours = np.arange(0.0, HORIZON_HOURS + 0.01, 1.0)
    q_gain_w = np.full_like(t_hours, q_step)
    t_out_c = np.full_like(t_hours, t_out_val)
    initial_state = np.array([t_out_val, t_out_val])

    y = _integrate_with(
        _broken_rc_derivatives_sign_flip, params, t_hours, q_gain_w, t_out_c, initial_state
    )
    t_i, t_e = y[0], y[1]

    relative_error = _energy_relative_error(params, t_hours, t_i, t_e, q_step, t_out_val)
    assert relative_error > ENERGY_RELATIVE_TOLERANCE, (
        "Expected the sign-flipped model to violate energy conservation, "
        f"but relative error was only {relative_error:.4f}"
    )

    direction = 1.0 if q_step > 0 else -1.0
    signed_diffs = direction * np.diff(t_i)
    assert signed_diffs.min() > -MONOTONICITY_TOLERANCE_C, (
        "Expected the sign-flipped model's runaway to stay monotonic "
        "(demonstrating monotonicity alone would NOT have caught this bug), "
        f"but min step was {signed_diffs.min():.4f} degC"
    )

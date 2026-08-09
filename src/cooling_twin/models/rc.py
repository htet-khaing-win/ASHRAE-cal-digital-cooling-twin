"""The 2R2C thermal model -- the physics core of the project.

Two state variables (indoor air temperature, envelope/thermal-mass
temperature) connected by two resistances and driven by two exogenous
inputs (heat gain, outdoor ambient temperature). `rc_derivatives()` is
the function every later stage depends on: L4.3 integrates it with
`solve_ivp`, M6 calibrates `RCParams` against real BDG2 load data, and
M7 explains what's left over once this physics is subtracted out.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class RCParams:
    """Physical parameters of the 2R2C zone model.

    Frozen so a parameter set, once constructed, cannot be silently
    mutated mid-simulation or mid-calibration -- M6's optimiser must
    build a *new* RCParams for every candidate it evaluates, never edit
    one in place. `__post_init__` enforces INV-5 (R > 0, C > 0) once,
    at construction time; nothing downstream needs to re-check it.

    Attributes:
        r_ie: Resistance between the indoor-air node and the
            envelope/thermal-mass node, K/W. Governs how fast heat
            picked up by the air reaches the structural mass (and vice
            versa) -- the *fast* coupling.
        r_ea: Resistance between the envelope/thermal-mass node and the
            outdoor ambient, K/W. This is where wall/roof insulation
            actually lives -- the *slow* coupling to outside.
        c_i: Thermal capacitance of the indoor air node, J/K. Air plus
            light contents (furniture, equipment) -- small, so this
            node responds quickly.
        c_e: Thermal capacitance of the envelope/thermal-mass node,
            J/K. Walls, slab, structural mass -- large, so this node
            responds slowly and smooths out fast swings in `c_i`.
    """

    r_ie: float
    r_ea: float
    c_i: float
    c_e: float

    def __post_init__(self) -> None:
        for name in ("r_ie", "r_ea", "c_i", "c_e"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(
                    f"INV-5 violated: {name}={value} -- thermal resistances "
                    "and capacitances must both be strictly positive."
                )


def rc_derivatives(
    t: float,
    state: np.ndarray,
    params: RCParams,
    q_gain: Callable[[float], float],
    t_out: Callable[[float], float],
) -> np.ndarray:
    """Time derivative of the 2R2C state vector, for use with an ODE solver.

    State-space form: dx/dt = f(t, x). This function computes only the
    derivative -- it does not integrate anything itself. That split is
    deliberate: `scipy.integrate.solve_ivp(fun, t_span, y0, args=...)`
    expects exactly this `fun(t, y, *args) -> dy/dt` signature (L4.3
    wires it up); keeping the physics here and the solver there means
    L4.4's model-order comparison and any future solver swap touch only
    the integration call, never this function.

    Governing equations (conservation of energy at each node):

        C_i * dT_i/dt = Q(t) - (T_i - T_e) / R_ie
        C_e * dT_e/dt = (T_i - T_e) / R_ie - (T_e - T_out(t)) / R_ea

    Args:
        t: Current simulation time, seconds. Passed through to `q_gain`
            and `t_out` so they can look up time-varying forcing (e.g.
            interpolated hourly BDG2 data in L4.3+); unused by the
            physics directly since R and C here are time-invariant.
        state: `[T_i, T_e]`, indoor air and envelope temperatures, degC.
        params: Calibratable physical parameters (INV-5 already
            enforced by `RCParams.__post_init__`).
        q_gain: Heat gain rate into the indoor-air node at time `t`, W.
            Positive gains warm the zone; a chiller's cooling effect
            enters as a negative value (L5.x).
        t_out: Outdoor ambient temperature at time `t`, degC.

    Returns:
        `[dT_i/dt, dT_e/dt]`, degC/s.
    """
    t_i, t_e = state
    q = q_gain(t)
    t_amb = t_out(t)

    d_t_i = (q - (t_i - t_e) / params.r_ie) / params.c_i
    d_t_e = ((t_i - t_e) / params.r_ie - (t_e - t_amb) / params.r_ea) / params.c_e

    return np.array([d_t_i, d_t_e])


def simulate(
    params: RCParams,
    t_hours: np.ndarray,
    q_gain_w: np.ndarray,
    t_out_c: np.ndarray,
    initial_state: np.ndarray,
    method: str = "RK45",
) -> pd.DataFrame:
    """Integrate the 2R2C model over a hourly forcing record.

    Wraps `scipy.integrate.solve_ivp` around `rc_derivatives()`. The
    solver works in continuous time and will ask `rc_derivatives()` for
    the derivative at times between the hourly samples (RK45 is
    adaptive-step, not fixed-step) -- `q_gain_w` and `t_out_c` are
    therefore linearly interpolated, not just indexed, before being
    handed to the solver.

    Args:
        params: Physical parameters (INV-5 enforced by `RCParams`).
        t_hours: Strictly increasing timestamps, hours, at least 2
            points. Need not be evenly spaced.
        q_gain_w: Heat gain into the indoor-air node at each `t_hours`
            timestamp, W. Same length as `t_hours`.
        t_out_c: Outdoor ambient temperature at each `t_hours`
            timestamp, degC. Same length as `t_hours`.
        initial_state: `[T_i0, T_e0]` at `t_hours[0]`, degC.
        method: `solve_ivp` integration method. RK45 (default, explicit
            Runge-Kutta 4(5), adaptive step) is the right choice for
            this model at realistic parameter values -- see L4.3's
            rationale for when that stops being true and an implicit
            method (`"Radau"`, `"BDF"`) would be needed instead.

    Returns:
        DataFrame indexed by `t_hours` (name `hours_since_start`) with
        columns `t_i` and `t_e`, degC.

    Raises:
        ValueError: If the input arrays have mismatched lengths, fewer
            than 2 points, or `t_hours` is not strictly increasing.
        RuntimeError: If `solve_ivp` fails to integrate (e.g. it hit
            its internal step-count limit) -- this is reported, never
            silently returned as a partial or wrong trajectory.
    """
    if not (len(t_hours) == len(q_gain_w) == len(t_out_c)):
        raise ValueError(
            f"t_hours ({len(t_hours)}), q_gain_w ({len(q_gain_w)}), and "
            f"t_out_c ({len(t_out_c)}) must be the same length -- one "
            "forcing value per timestamp."
        )
    if len(t_hours) < 2:
        raise ValueError("t_hours must have at least 2 points to integrate over")
    if not np.all(np.diff(t_hours) > 0):
        raise ValueError("t_hours must be strictly increasing")

    t_seconds = (t_hours - t_hours[0]) * _SECONDS_PER_HOUR

    def q_gain(t: float) -> float:
        return float(np.interp(t, t_seconds, q_gain_w))

    def t_out(t: float) -> float:
        return float(np.interp(t, t_seconds, t_out_c))

    result = solve_ivp(
        fun=rc_derivatives,
        t_span=(t_seconds[0], t_seconds[-1]),
        y0=np.asarray(initial_state, dtype=float),
        method=method,
        t_eval=t_seconds,
        args=(params, q_gain, t_out),
    )

    if not result.success:
        raise RuntimeError(f"solve_ivp failed to integrate: {result.message}")

    return pd.DataFrame(
        {"t_i": result.y[0], "t_e": result.y[1]},
        index=pd.Index(t_hours, name="hours_since_start"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Illustrative parameters -- NOT yet calibrated (that's M6). Same
    # order of magnitude as L4.1's single R_ENVELOPE/C_ZONE, now split
    # across two nodes: r_ie+r_ea ~= L4.1's R_ENVELOPE, c_e ~= L4.1's
    # C_ZONE (the envelope dominates total thermal mass), c_i is the
    # new, smaller, fast-responding air node.
    demo_params = RCParams(r_ie=2e-4, r_ea=5e-4, c_i=5e6, c_e=3e7)
    demo_state = np.array([30.0, 30.0])  # zone and envelope both start at ambient

    derivs = rc_derivatives(
        t=0.0,
        state=demo_state,
        params=demo_params,
        q_gain=lambda _t: 50_000.0,
        t_out=lambda _t: 30.0,
    )
    logger.info(
        "at t=0 (T_i=T_e=T_out=30degC): dT_i/dt=%.6f degC/s, dT_e/dt=%.6f degC/s",
        derivs[0],
        derivs[1],
    )

    # Validate simulate() against L4.1's closed-form 1R1C solution. As
    # r_ie -> 0, T_i and T_e are forced together (INV-8: the r_ie terms
    # cancel exactly when the two node equations are summed), and the
    # 2R2C system degenerates to 1R1C with R=r_ea, C=c_i+c_e. There is
    # no independent way to check "is this solver right" from inside
    # this file -- this limiting case is the only ground truth available
    # before real BDG2 data and M6 calibration exist.
    #
    # r_ie=1e-6, not smaller: pushing it toward the true r_ie->0 limit
    # (tried 1e-8 while writing this) makes RK45 (explicit, adaptive)
    # hang -- the fast air-envelope mode becomes so much faster than the
    # 24h window that RK45 needs an impractical number of tiny steps to
    # stay stable. That failure mode IS L4.3's stiffness topic; see the
    # "why it's written this way" section for the fix (an implicit
    # method) rather than chasing r_ie -> 0 with RK45 here.
    validation_params = RCParams(r_ie=1e-6, r_ea=5e-4, c_i=5e6, c_e=3e7)
    t_hours = np.linspace(0, 24, 500)
    q_step_w = 50_000.0
    q_gain_w = np.full_like(t_hours, q_step_w)
    t_out_c = np.full_like(t_hours, 30.0)

    sim = simulate(validation_params, t_hours, q_gain_w, t_out_c, np.array([30.0, 30.0]))

    r_equiv, c_equiv = validation_params.r_ea, validation_params.c_i + validation_params.c_e
    tau_equiv_s = r_equiv * c_equiv
    analytic_t_i = 30.0 + q_step_w * r_equiv * (
        1 - np.exp(-t_hours * _SECONDS_PER_HOUR / tau_equiv_s)
    )
    max_error = np.max(np.abs(sim["t_i"].to_numpy() - analytic_t_i))
    logger.info(
        "simulate() vs L4.1 closed-form (r_ie->0 limit, tau=%.2fh): "
        "max |T_i error| = %.2e degC",
        tau_equiv_s / _SECONDS_PER_HOUR,
        max_error,
    )

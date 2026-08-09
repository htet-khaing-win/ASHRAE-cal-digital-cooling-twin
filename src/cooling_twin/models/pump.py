"""Pump power -- the affinity laws (P ~ flow^3).

A centrifugal pump's affinity laws relate speed N to flow, head, and
power: flow ~ N, head ~ N^2, power ~ N^3. `pump_power()` uses the last
of these to model a variable-speed (VFD) pump; a fixed-speed pump gets
its own branch because it does NOT get this scaling for free (see
`pump_power()`'s docstring for why). M6 calibrates `PumpParams` against
real BDG2 electricity data; this module is what turns L1.4's "2x flow =
8x power" rule of thumb into code M5's plant assembly (L5.5) can call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_AFFINITY_LAW_POWER_EXPONENT = 3  # int, not 3.0 -- float**float resolves to Any in typeshed


@dataclass(frozen=True)
class PumpParams:
    """One pump's rated operating point.

    Frozen for the same reason as `RCParams` (L4.2) and `ChillerCurves`
    (L5.1) -- M6's optimiser builds a new `PumpParams` per candidate,
    never mutates one mid-calibration.

    Attributes:
        flow_rated_m3_per_s: Rated (100% speed) volumetric flow rate,
            m^3/s. Must be > 0.
        power_rated_kw: Electric power draw at rated flow and rated
            speed, kW. Must be > 0.
    """

    flow_rated_m3_per_s: float
    power_rated_kw: float

    def __post_init__(self) -> None:
        if self.flow_rated_m3_per_s <= 0:
            raise ValueError(
                f"flow_rated_m3_per_s must be > 0, got {self.flow_rated_m3_per_s}"
            )
        if self.power_rated_kw <= 0:
            raise ValueError(f"power_rated_kw must be > 0, got {self.power_rated_kw}")


def pump_power(
    flow_m3_per_s: float,
    params: PumpParams,
    variable_speed: bool,
) -> float:
    """Pump electric power at a given flow rate.

    Two structurally different ways to run a pump below its rated flow,
    with very different power consequences:

    - **Variable speed (VFD):** motor speed N drops with flow
      (flow ~ N), so the affinity laws give power ~ N^3 ~ flow^3.
      Halving flow cuts power to 1/8 -- L1.4's "2x flow = 8x power"
      rule of thumb, run in reverse.
    - **Fixed speed:** the motor always runs at rated speed; flow is
      reduced by throttling a control valve instead. The pump still
      does almost the same work fighting the valve's added resistance,
      so power stays close to `power_rated_kw` regardless of flow. This
      is exactly why VFDs are worth installing, and exactly why a
      constant-speed system does NOT get the flow^3 saving for free --
      the saving comes from the speed change, not from the flow change
      itself.

    Args:
        flow_m3_per_s: Actual volumetric flow rate, m^3/s. Must be in
            `[0, params.flow_rated_m3_per_s]` -- a pump cannot exceed
            the flow its rated (100%) speed produces.
        params: The pump's rated operating point.
        variable_speed: `True` for a VFD-controlled pump (affinity-law
            scaling); `False` for a fixed-speed pump with valve
            throttling (power stays at `power_rated_kw` for any valid
            flow).

    Returns:
        Electric power draw, kW.

    Raises:
        ValueError: If `flow_m3_per_s` is negative, or exceeds
            `params.flow_rated_m3_per_s` (the flow being asked for
            would require exceeding rated speed).
    """
    if flow_m3_per_s < 0:
        raise ValueError(f"flow_m3_per_s must be >= 0, got {flow_m3_per_s}")
    if flow_m3_per_s > params.flow_rated_m3_per_s:
        raise ValueError(
            f"flow_m3_per_s={flow_m3_per_s:.5f} exceeds "
            f"flow_rated_m3_per_s={params.flow_rated_m3_per_s:.5f} -- this "
            "pump cannot deliver more flow than its rated speed produces."
        )

    if not variable_speed:
        return params.power_rated_kw

    flow_ratio = flow_m3_per_s / params.flow_rated_m3_per_s
    return params.power_rated_kw * flow_ratio**_AFFINITY_LAW_POWER_EXPONENT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Standard water properties (physical constants, not a
    # 03_DOMAIN_REFERENCE.md fact -- these don't vary by building).
    RHO_WATER_KG_PER_M3 = 1000.0
    CP_WATER_J_PER_KG_K = 4186.0

    q_load_kw = 1000.0  # same illustrative load scale as L5.1/L5.2's demos
    t_delta_design_k = 6.0  # within 03_DOMAIN_REFERENCE.md SS1's 5-7K design range
    t_delta_degraded_k = 5.5  # a MODEST slip, not a dramatic one -- see below
    t_delta_severe_k = 5.0

    def _flow_for_load(q_kw: float, delta_t_k: float) -> float:
        """Q = rho * cp * flow * deltaT, solved for flow."""
        return (q_kw * 1000.0) / (RHO_WATER_KG_PER_M3 * CP_WATER_J_PER_KG_K * delta_t_k)

    flow_design = _flow_for_load(q_load_kw, t_delta_design_k)
    flow_degraded = _flow_for_load(q_load_kw, t_delta_degraded_k)

    # Pump sized for the WORSE case (degraded delta-T) -- realistic
    # sizing practice, and what makes flow_degraded land exactly at
    # the rated ceiling below. power_rated_kw=75.0 is an illustrative
    # CHW-pump-scale choice, not a fitted/verified number.
    demo_params = PumpParams(flow_rated_m3_per_s=flow_degraded, power_rated_kw=75.0)
    logger.info(
        "pump rated flow=%.2f L/s (sized for the %.1fK worse-case delta-T)",
        demo_params.flow_rated_m3_per_s * 1000.0,
        t_delta_degraded_k,
    )

    # L1.4's rule of thumb, exactly: half of RATED flow -> 1/8 of rated power.
    power_half_flow = pump_power(
        demo_params.flow_rated_m3_per_s / 2, demo_params, variable_speed=True
    )
    logger.info(
        "VFD at 50%% of rated flow: power=%.3f kW (rated/8=%.3f kW)",
        power_half_flow,
        demo_params.power_rated_kw / 8,
    )

    # Healthy design-delta-T operation: flow sits BELOW the rated
    # ceiling, so a VFD only needs partial speed -- but a fixed-speed
    # pump still burns full rated power throttling a valve to get there.
    power_vfd_design = pump_power(flow_design, demo_params, variable_speed=True)
    power_fixed_design = pump_power(flow_design, demo_params, variable_speed=False)
    logger.info(
        "design delta-T=%.1fK: flow=%.2f L/s -- VFD power=%.2f kW, "
        "fixed-speed power=%.2f kW (fixed-speed wastes %.2f kW at the valve)",
        t_delta_design_k,
        flow_design * 1000.0,
        power_vfd_design,
        power_fixed_design,
        power_fixed_design - power_vfd_design,
    )

    # Low-delta-T syndrome (03_DOMAIN_REFERENCE.md SS5 fault taxonomy):
    # delta-T slips from 6.0K design to 5.5K. Flow rises to hold Q
    # constant, landing exactly at this pump's rated ceiling.
    power_vfd_degraded = pump_power(flow_degraded, demo_params, variable_speed=True)
    logger.info(
        "delta-T slips %.1fK->%.1fK: flow %.2f->%.2f L/s, VFD power %.2f->%.2f kW "
        "(+%.1f%%, matches the documented 10-30%% low-delta-T pump energy cost)",
        t_delta_design_k,
        t_delta_degraded_k,
        flow_design * 1000.0,
        flow_degraded * 1000.0,
        power_vfd_design,
        power_vfd_degraded,
        (power_vfd_degraded / power_vfd_design - 1) * 100,
    )

    # Push delta-T lower still: required flow now exceeds what this
    # pump can deliver even at 100% speed -- the real "riding the curve
    # at max speed, can't meet demand" symptom of low-delta-T syndrome.
    flow_severe = _flow_for_load(q_load_kw, t_delta_severe_k)
    try:
        pump_power(flow_severe, demo_params, variable_speed=True)
    except ValueError as exc:
        logger.info(
            "delta-T=%.1fK needs %.2f L/s -- pump correctly rejected it: %s",
            t_delta_severe_k,
            flow_severe * 1000.0,
            exc,
        )

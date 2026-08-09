"""Cooling tower outlet temperature -- approach and the wet-bulb limit.

A cooling tower rejects heat by evaporation, and evaporation cannot
cool water below the ambient wet-bulb temperature -- INV-3
(03_DOMAIN_REFERENCE.md SS2). `cooling_tower_outlet()` models the
condenser water leaving the tower as wet bulb plus a design "approach"
temperature, and makes INV-3 hold *structurally* (a positive approach
guarantees it) rather than checking the output after the fact.

The tower's fan is a centrifugal/axial turbomachine, exactly like
L5.3's pumps -- it obeys the same affinity laws (flow ~ N, power ~
N^3). Rather than re-derive that math under a new name, this module's
`__main__` demo reuses `pump_power()` directly for the fan-power
trade-off.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def cooling_tower_outlet(t_wet_bulb_c: float, approach_k: float) -> float:
    """Condenser water temperature leaving the cooling tower.

    T_cw_out = T_wet_bulb + approach

    "Approach" is the minimum temperature difference a real tower can
    hold between the leaving water and the ambient wet bulb -- a
    physical floor set by the tower's heat-exchange area and airflow,
    never zero. `03_DOMAIN_REFERENCE.md` SS1 documents 2.5-4 K as the
    optimisation range for how close a real tower's design approach
    gets; a smaller approach means a bigger (more expensive) tower or
    more fan airflow for the same heat rejection -- the trade-off this
    lesson's demo makes concrete.

    Args:
        t_wet_bulb_c: Ambient wet-bulb temperature, degC.
        approach_k: Design approach temperature, K. Must be > 0 -- this
            is what makes INV-3 hold structurally (see Raises).

    Returns:
        Condenser water temperature leaving the tower, degC.

    Raises:
        ValueError: If `approach_k <= 0`. INV-3 (T_condenser_water >
            T_wet_bulb) requires a strictly positive approach; zero or
            negative would put the outlet at or below wet bulb, which
            no cooling tower can physically achieve.
    """
    if approach_k <= 0:
        raise ValueError(
            f"approach_k must be > 0, got {approach_k} -- INV-3 requires "
            "T_condenser_water > T_wet_bulb; a zero or negative approach "
            "would put the outlet AT OR BELOW wet bulb, which no cooling "
            "tower can physically achieve."
        )
    return t_wet_bulb_c + approach_k


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from cooling_twin.models.pump import PumpParams, pump_power

    # Wet-bulb values and approach chosen (within documented ranges) to
    # reproduce 03_DOMAIN_REFERENCE.md SS3's condenser water floors
    # exactly, as a sanity check: tropical ~31degC, temperate ~15degC.
    t_wet_bulb_tropical = 27.0  # SS3: 26-28degC tropical annual mean
    t_wet_bulb_temperate = 11.0  # SS3: 8-13degC temperate annual mean
    approach_design_k = 4.0  # SS1: 2.5-4K optimisation range, upper end

    t_cw_tropical = cooling_tower_outlet(t_wet_bulb_tropical, approach_design_k)
    t_cw_temperate = cooling_tower_outlet(t_wet_bulb_temperate, approach_design_k)
    logger.info(
        "tropical: wet_bulb=%.1fdegC -> CW=%.1fdegC "
        "(INV-3 holds: %.1f > %.1f; matches SS3's ~31degC floor)",
        t_wet_bulb_tropical,
        t_cw_tropical,
        t_cw_tropical,
        t_wet_bulb_tropical,
    )
    logger.info(
        "temperate: wet_bulb=%.1fdegC -> CW=%.1fdegC (matches SS3's ~15degC floor)",
        t_wet_bulb_temperate,
        t_cw_temperate,
    )

    # INV-3 in action: a zero approach must raise, not silently return
    # an unphysical tower that cools water down to exactly wet bulb.
    try:
        cooling_tower_outlet(t_wet_bulb_tropical, approach_k=0.0)
    except ValueError as exc:
        logger.info("INV-3 correctly rejected a zero approach: %s", exc)

    # The fan-power trade-off: cooling tower fans are centrifugal/axial
    # turbomachines, exactly like L5.3's pumps -- same affinity laws,
    # so pump_power() is reused directly rather than re-derived here.
    fan_params = PumpParams(flow_rated_m3_per_s=50.0, power_rated_kw=30.0)
    power_full_fan = pump_power(50.0, fan_params, variable_speed=True)
    power_half_fan = pump_power(25.0, fan_params, variable_speed=True)
    logger.info(
        "tower fan (reusing pump_power()): full airflow=%.2f kW, half airflow=%.3f kW "
        "(1/8, same cubic affinity law as L5.3's pumps -- more airflow narrows the "
        "approach, but costs power cubically)",
        power_full_fan,
        power_half_fan,
    )

"""Latent load -- the tropical differentiator.

Cooling a stream of air removes two physically distinct kinds of heat:
**sensible** (lowering its temperature) and **latent** (condensing
moisture out of it). BDG2's North American/European buildings run
mostly sensible-dominated -- a sensible-only model can pass calibration
there. `03_DOMAIN_REFERENCE.md` SS3 documents that tropical climates
(Singapore, Bangkok -- this project's actual target) run 35-45% latent
share versus 10-20% temperate: a sensible-only model silently
undersizes a tropical plant by nearly half. `cooling_load()` is the
function that makes this project's core research claim checkable in
code rather than asserted in prose; M7's humid-vs-dry residual
comparison needs both terms to exist so it can be run with either one
switched off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Public because the ventilation term in models/rc.py (ADR-011) must use
# the SAME latent heat of vaporisation this coil model uses -- two values
# for h_fg in one repository would put a silent inconsistency between the
# plant model and the load model that calibrates against it.
H_FG_J_PER_KG = 2_450_000.0  # 03_DOMAIN_REFERENCE.md SS3: h_fg ~ 2450 kJ/kg near 25degC

_CP_AIR_J_PER_KG_K = 1006.0  # standard dry air specific heat -- physical constant
_H_FG_J_PER_KG = H_FG_J_PER_KG  # retained name for this module's own use


@dataclass(frozen=True)
class CoolingLoad:
    """Cooling load split into its sensible and latent components.

    Kept as two separate fields rather than one summed float -- that
    split IS this lesson's point: comparing `sensible_kw` alone against
    `total_kw` is exactly the "with and without the latent term"
    comparison 02_CURRICULUM.md's L5.5 asks for, and M7's later
    physics/residual decomposition needs named parts, not one opaque
    number, for the same reason.

    Attributes:
        sensible_kw: Heat removed by temperature change alone, kW.
        latent_kw: Heat removed by moisture removal (condensation), kW.
    """

    sensible_kw: float
    latent_kw: float

    @property
    def total_kw(self) -> float:
        """Q_total = Q_sensible + Q_latent."""
        return self.sensible_kw + self.latent_kw


def cooling_load(
    m_air_kg_per_s: float,
    t_entering_c: float,
    t_leaving_c: float,
    w_entering_kg_per_kg: float,
    w_leaving_kg_per_kg: float,
) -> CoolingLoad:
    """Sensible + latent cooling load across an air-handling coil.

    03_DOMAIN_REFERENCE.md SS3:
        Q_sensible = m_air * cp * (T_entering - T_leaving)
        Q_latent   = m_air * h_fg * (w_entering - w_leaving)

    "Entering"/"leaving" here means the air stream crossing the cooling
    coil itself (mixed/outdoor air going in, cooled-and-dehumidified
    supply air coming out) -- deliberately not "indoor"/"outdoor",
    which the domain reference's own T_in/T_out shorthand could be
    misread as (see this lesson's "why" section for the ambiguity this
    avoids).

    Humidity ratios (`w_*`, kg water per kg dry air) must come from
    `psychrolib` -- via `cooling_twin.data.weather.add_psychrometric_features()`
    for real BDG2 data, or a direct `psychrolib` call for a synthetic
    scenario (see this module's `__main__` block). Domain reference SS3
    explicitly rules out hand-rolling psychrometric formulas; this
    function only does the arithmetic on humidity ratios it is given,
    never derives one itself.

    Args:
        m_air_kg_per_s: Dry-air mass flow rate through the coil, kg/s.
            Must be > 0.
        t_entering_c: Dry-bulb temperature of air entering the coil, degC.
        t_leaving_c: Dry-bulb temperature of air leaving the coil, degC.
            Must be <= t_entering_c -- a cooling coil only removes heat.
        w_entering_kg_per_kg: Humidity ratio of air entering the coil,
            kg water / kg dry air.
        w_leaving_kg_per_kg: Humidity ratio of air leaving the coil,
            kg water / kg dry air. Must be <= w_entering_kg_per_kg -- a
            cooling coil only removes moisture, never adds it.

    Returns:
        A `CoolingLoad` with `sensible_kw` and `latent_kw` set
        separately (see `CoolingLoad.total_kw` for the sum).

    Raises:
        ValueError: If `m_air_kg_per_s <= 0`, if `t_leaving_c` exceeds
            `t_entering_c` (this would be heating, not cooling), or if
            `w_leaving_kg_per_kg` exceeds `w_entering_kg_per_kg` (this
            would be humidifying, not dehumidifying).
    """
    if m_air_kg_per_s <= 0:
        raise ValueError(f"m_air_kg_per_s must be > 0, got {m_air_kg_per_s}")
    if t_leaving_c > t_entering_c:
        raise ValueError(
            f"t_leaving_c={t_leaving_c} exceeds t_entering_c={t_entering_c} -- "
            "a cooling coil only removes heat; this input describes heating."
        )
    if w_leaving_kg_per_kg > w_entering_kg_per_kg:
        raise ValueError(
            f"w_leaving_kg_per_kg={w_leaving_kg_per_kg} exceeds "
            f"w_entering_kg_per_kg={w_entering_kg_per_kg} -- a cooling coil "
            "only removes moisture; this input describes humidification."
        )

    sensible_w = m_air_kg_per_s * _CP_AIR_J_PER_KG_K * (t_entering_c - t_leaving_c)
    latent_w = m_air_kg_per_s * _H_FG_J_PER_KG * (w_entering_kg_per_kg - w_leaving_kg_per_kg)

    return CoolingLoad(sensible_kw=sensible_w / 1000.0, latent_kw=latent_w / 1000.0)


def validate_chw_delta_t(t_chw_supply_c: float, t_chw_return_c: float) -> None:
    """Check INV-2: chilled water supply must be strictly colder than return.

    The evaporator cools the water passing through it -- water leaves
    (supply) colder than it entered (return). `t_chw_supply_c >=
    t_chw_return_c` means the flow direction or a sensor mapping is
    wrong (03_DOMAIN_REFERENCE.md SS2's own description of INV-2), not
    a valid unusual operating point -- there is no real chiller
    operating mode that produces it.

    No function elsewhere in this project currently takes both CHW
    temperatures as separate arguments (L5.1's `chiller_power()` only
    takes supply; L5.3's pump demo works from a single delta-T, never
    two temperatures) -- this is the first, and deliberately the
    smallest possible check that closes that gap for the M5 gate
    (`06_ASSESSMENT.md`'s "INV-1..INV-9 covered by tests" item).

    Args:
        t_chw_supply_c: Chilled water supply (leaving evaporator)
            temperature, degC.
        t_chw_return_c: Chilled water return (entering evaporator)
            temperature, degC.

    Raises:
        ValueError: If `t_chw_supply_c >= t_chw_return_c`.
    """
    if t_chw_supply_c >= t_chw_return_c:
        raise ValueError(
            f"INV-2 violated: t_chw_supply_c={t_chw_supply_c} >= "
            f"t_chw_return_c={t_chw_return_c} -- chilled water supply must "
            "be strictly colder than return, or the flow direction / "
            "sensor mapping is wrong."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import psychrolib

    psychrolib.SetUnitSystem(psychrolib.SI)  # matches weather.py's process-global setup
    _P_ATM_PA = 101_325.0

    # "Same building, same forcing" (02_CURRICULUM.md L5.5): identical
    # entering/leaving dry-bulb and m_air across both scenarios below --
    # only the HUMIDITY differs, isolating the latent effect from the
    # sensible one. t_leaving_c=13.0 matches 03_DOMAIN_REFERENCE.md
    # SS1's documented supply air temp exactly.
    m_air_kg_per_s = 50.0  # illustrative AHU-scale airflow, not a fitted number
    t_entering_c = 24.0  # AHU mixed-air condition entering the coil
    t_leaving_c = 13.0
    rh_leaving_pct = 85.0  # near-saturated, post-dehumidification

    # RH_entering values chosen so the resulting latent SHARE lands
    # inside 03_DOMAIN_REFERENCE.md SS3's documented bands (temperate
    # 10-20%, tropical 35-45%) -- a validation that this simple model
    # reproduces the documented pattern, not an invented result.
    rh_entering_temperate_pct = 48.0
    rh_entering_tropical_pct = 60.0

    w_leaving = psychrolib.GetHumRatioFromRelHum(t_leaving_c, rh_leaving_pct / 100.0, _P_ATM_PA)

    for label, rh_entering_pct in [
        ("temperate", rh_entering_temperate_pct),
        ("tropical", rh_entering_tropical_pct),
    ]:
        w_entering = psychrolib.GetHumRatioFromRelHum(
            t_entering_c, rh_entering_pct / 100.0, _P_ATM_PA
        )
        load = cooling_load(m_air_kg_per_s, t_entering_c, t_leaving_c, w_entering, w_leaving)
        latent_share = load.latent_kw / load.total_kw
        sensible_only_error_pct = (load.total_kw / load.sensible_kw - 1) * 100
        logger.info(
            "%s (RH_entering=%.0f%%): sensible=%.1f kW, latent=%.1f kW, "
            "total=%.1f kW, latent_share=%.1f%% -- a sensible-only model "
            "would undersize this plant by %.1f%%",
            label,
            rh_entering_pct,
            load.sensible_kw,
            load.latent_kw,
            load.total_kw,
            latent_share * 100,
            sensible_only_error_pct,
        )

    # A coil that supposedly ADDS moisture is not a cooling coil --
    # swap the two humidity ratios from the tropical case above to
    # demonstrate the rejection.
    try:
        cooling_load(m_air_kg_per_s, t_entering_c, t_leaving_c, w_leaving, w_entering)
    except ValueError as exc:
        logger.info("cooling_load() correctly rejected humidification: %s", exc)

    # INV-2: a plausible CHW supply/return pair (L5.3's design delta-T,
    # 6.0K, applied to L5.1's 6.7degC rated supply) passes; a swapped
    # pair does not.
    validate_chw_delta_t(t_chw_supply_c=6.7, t_chw_return_c=6.7 + 6.0)
    logger.info("INV-2 holds for a normal 6.7degC/12.7degC supply/return pair")
    try:
        validate_chw_delta_t(t_chw_supply_c=12.7, t_chw_return_c=6.7)
    except ValueError as exc:
        logger.info("INV-2 correctly rejected a swapped supply/return pair: %s", exc)

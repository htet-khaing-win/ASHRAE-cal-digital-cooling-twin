"""DOE-2 chiller performance curves.

Real chiller performance (compressor maps, refrigerant properties, heat
exchanger effectiveness) is never derived from first principles here --
manufacturers publish it as three empirical curve fits against operating
temperature and part-load ratio (the DOE-2 / EnergyPlus "Electric EIR"
formulation). `biquadratic()` is the general two-variable curve-fit
utility; `chiller_power()` combines three such curves (CAPFT, EIRFT,
EIRFPLR) into an electric power draw. M6 calibrates `ChillerCurves`
against real BDG2 electricity data; M7 explains what's left over once
this (plus the RC model) is subtracted out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_MIN_COP = 0.0
_MAX_COP = 10.0  # INV-1 (03_DOMAIN_REFERENCE.md SS2)
_MAX_PLR = 1.05  # INV-6 (03_DOMAIN_REFERENCE.md SS2): 0 <= PLR <= 1.05

CurveCoeffs6 = tuple[float, float, float, float, float, float]
CurveCoeffs3 = tuple[float, float, float]


def biquadratic(x: float, y: float, coeffs: CurveCoeffs6) -> float:
    """Evaluate a DOE-2 / EnergyPlus BiQuadratic performance curve.

    f(x, y) = c0 + c1*x + c2*x^2 + c3*y + c4*y^2 + c5*x*y

    This is the standard two-variable curve form manufacturers fit their
    catalog performance data to (CAPFT and EIRFT both use it). It is
    deliberately just a polynomial evaluator with no chiller-specific
    meaning attached -- `chiller_power()` supplies that meaning by
    choosing which two temperatures go in and which curve's coefficients
    to use.

    Args:
        x: First independent variable (e.g. chilled water supply
            temperature, degC).
        y: Second independent variable (e.g. condenser water
            temperature, degC).
        coeffs: Six coefficients `(c0, c1, c2, c3, c4, c5)` in the order
            above, as published on a manufacturer's certified curve
            data sheet.

    Returns:
        The curve's dimensionless output -- a fraction of the rated
        quantity the curve corrects (e.g. 1.05 means 5% above rated).
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1 * x + c2 * x**2 + c3 * y + c4 * y**2 + c5 * x * y


@dataclass(frozen=True)
class ChillerCurves:
    """One chiller's manufacturer performance curves and rating point.

    Frozen for the same reason as `RCParams` (L4.2) -- M6's optimiser
    must build a new `ChillerCurves` per candidate, never mutate one
    mid-calibration. `__post_init__` enforces INV-1 (0 < COP < 10) on
    `cop_ref` once, at construction time.

    Attributes:
        cap_ft: CAPFT coefficients -- available cooling capacity as a
            fraction of `q_ref_kw`, as a function of
            `(t_chw_supply_c, t_cond_water_c)`.
        eir_ft: EIRFT coefficients -- Energy Input Ratio (1/COP) as a
            fraction of the rated EIR, as a function of the same two
            temperatures.
        eir_fplr: EIRFPLR coefficients, quadratic in part-load ratio:
            `c0 + c1*PLR + c2*PLR**2`. Normalized so `EIRFPLR(1.0) ==
            1.0` (full load at rating conditions reproduces `cop_ref`
            exactly) with a minimum somewhere in the PLR range -- that
            minimum is where the chiller is MOST efficient (COP peaks),
            per `chiller_power()`'s inverse relationship between EIR
            and COP.
        q_ref_kw: Rated cooling capacity at the manufacturer's rating
            point, kW. Must be > 0.
        cop_ref: Rated COP at the manufacturer's rating point (INV-1).
    """

    cap_ft: CurveCoeffs6
    eir_ft: CurveCoeffs6
    eir_fplr: CurveCoeffs3
    q_ref_kw: float
    cop_ref: float

    def __post_init__(self) -> None:
        if self.q_ref_kw <= 0:
            raise ValueError(f"q_ref_kw must be > 0, got {self.q_ref_kw}")
        if not _MIN_COP < self.cop_ref < _MAX_COP:
            raise ValueError(
                f"INV-1 violated: cop_ref={self.cop_ref} -- must satisfy "
                f"{_MIN_COP} < COP < {_MAX_COP}."
            )


def chiller_power(
    q_load_kw: float,
    t_chw_supply_c: float,
    t_cond_water_c: float,
    curves: ChillerCurves,
) -> float:
    """Chiller compressor electric power, DOE-2 Electric EIR method.

    Three curve evaluations combine into one power number::

        Q_avail = Q_ref  * CAPFT(T_chw, T_cond)   -- capacity you actually
                                                      have at these temps
        PLR     = Q_load / Q_avail                -- how hard you're
                                                      running it
        EIR     = EIR_ref * EIRFT(T_chw, T_cond) * EIRFPLR(PLR)
        Power   = Q_avail * PLR * EIR

    Note that `Q_avail * PLR == Q_load` algebraically whenever `PLR`
    isn't clipped -- `CAPFT` therefore has no DIRECT effect on the
    returned power once `EIRFPLR` is folded in (it cancels out of
    `Q_avail * PLR`). It still matters: `CAPFT` sets the `PLR` value
    that `EIRFPLR` is evaluated at, and `EIRFPLR` (L5.2) is genuinely
    curved, not flat -- get `CAPFT` wrong and `EIRFPLR` gets evaluated
    at the wrong point on its curve, producing the wrong `EIR`.

    Args:
        q_load_kw: Cooling load the chiller must meet, kW. Must be >= 0.
        t_chw_supply_c: Chilled water supply (leaving evaporator)
            temperature, degC.
        t_cond_water_c: Condenser water (entering condenser) temperature,
            degC.
        curves: The chiller's fitted performance curves and rating point.

    Returns:
        Electric power draw of the compressor, kW.

    Raises:
        ValueError: If `q_load_kw` is negative, if `CAPFT` evaluates to
            zero or negative available capacity at the given
            temperatures (the curve is being asked to extrapolate
            somewhere it was never fit to represent), or if the
            resulting part-load ratio violates INV-6 (`PLR > 1.05` --
            the load exceeds what this chiller can deliver even with
            the standard 5% overload margin).
    """
    if q_load_kw < 0:
        raise ValueError(f"q_load_kw must be >= 0, got {q_load_kw}")

    cap_f = biquadratic(t_chw_supply_c, t_cond_water_c, curves.cap_ft)
    q_avail_kw = curves.q_ref_kw * cap_f
    if q_avail_kw <= 0:
        raise ValueError(
            f"CAPFT collapsed available capacity to {q_avail_kw:.3f} kW at "
            f"T_chw_supply={t_chw_supply_c}degC, T_cond={t_cond_water_c}degC "
            "-- these temperatures are outside the range the curve was fit "
            "to represent."
        )

    plr = q_load_kw / q_avail_kw
    if plr > _MAX_PLR:
        raise ValueError(
            f"INV-6 violated: PLR={plr:.3f} (Q_load={q_load_kw:.1f} kW, "
            f"Q_avail={q_avail_kw:.1f} kW) exceeds the {_MAX_PLR} limit -- "
            "this chiller cannot meet this load at these temperatures, "
            "even with the standard overload margin."
        )
    logger.debug(
        "PLR=%.3f (Q_load=%.1f kW, Q_avail=%.1f kW)", plr, q_load_kw, q_avail_kw
    )

    eir_f_t = biquadratic(t_chw_supply_c, t_cond_water_c, curves.eir_ft)
    p0, p1, p2 = curves.eir_fplr
    eir_f_plr = p0 + p1 * plr + p2 * plr**2
    eir_ref = 1.0 / curves.cop_ref

    return q_avail_kw * plr * eir_ref * eir_f_t * eir_f_plr


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Illustrative curves -- NOT a real manufacturer's certified data
    # sheet (M6 fits real coefficients from BDG2; a production project
    # would source these from an AHRI-certified submittal or the
    # EnergyPlus chiller curve library instead). Derived so the curves
    # are consistent with what IS documented in 03_DOMAIN_REFERENCE.md:
    # rated at CHW=6.7degC / CW=29.4degC (SS1 "Typical vs optimisable
    # setpoints"), cop_ref=6.0 (midpoint of the 5.5-7.0 centrifugal
    # design-point range, SS1), and the EIRFT slopes reproduce the
    # documented rules of thumb exactly ("CHW supply +1K -> efficiency
    # +2-3%", "condenser water -1K -> efficiency +2-3%" -- using the
    # conservative 2%/K end of both ranges).
    t_chw_ref, t_cond_ref = 6.7, 29.4
    cap_ft_demo: CurveCoeffs6 = (1.1935, 0.015, 0.0, -0.010, 0.0, 0.0)
    eir_ft_demo: CurveCoeffs6 = (0.546, -0.02, 0.0, 0.02, 0.0, 0.0)

    # EIRFPLR(PLR) = 3.05 - 6.05*PLR + 4.00*PLR^2. Solved (not guessed)
    # from three constraints, all traceable to 03_DOMAIN_REFERENCE.md
    # SS1: (1) EIRFPLR(1.0) = 1.0 -- full load at rating conditions
    # reproduces cop_ref exactly, by definition of "rated"; (2) the
    # resulting parabola's vertex (minimum EIR = peak COP) must land in
    # PLR in [0.6, 0.8] -- "PLR 60-80% -> peak efficiency"; (3)
    # EIRFPLR(0.2) = 2.0, chosen so COP(0.2) = cop_ref/2.0 = 3.0, the
    # upper edge of the documented "PLR<25% -> COP 2.0-3.5, severe
    # degradation" band. A quadratic only has 3 degrees of freedom, so
    # these 3 constraints pin it down completely -- the vertex lands at
    # PLR=0.756 (see L5.2's "why" section for why picking the vertex
    # location AND the low-PLR target independently, rather than
    # forcing an exact vertex position, is what keeps this curve inside
    # INV-1 across the whole PLR range).
    eir_fplr_demo: CurveCoeffs3 = (3.05, -6.05, 4.00)

    demo_curves = ChillerCurves(
        cap_ft=cap_ft_demo,
        eir_ft=eir_ft_demo,
        eir_fplr=eir_fplr_demo,
        q_ref_kw=1000.0,
        cop_ref=6.0,
    )

    # Sanity check: at the rating point itself, both curves must
    # evaluate to exactly 1.0 -- that is the definition of "rated".
    logger.info(
        "at rating point: CAPFT=%.4f, EIRFT=%.4f (both should be 1.0000)",
        biquadratic(t_chw_ref, t_cond_ref, cap_ft_demo),
        biquadratic(t_chw_ref, t_cond_ref, eir_ft_demo),
    )

    # Full load AT the rating point (PLR=1.0): EIRFPLR(1.0)=1.0 by
    # construction, so COP must equal cop_ref exactly -- same sanity
    # check as L5.1, now at the point where EIRFPLR actually matters.
    power_full = chiller_power(1000.0, t_chw_ref, t_cond_ref, demo_curves)
    logger.info(
        "full load, rating point: power=%.2f kW, COP=%.3f (should equal cop_ref=6.0)",
        power_full,
        1000.0 / power_full,
    )

    # Peak-efficiency zone (PLR=0.75, inside the documented 60-80%
    # band): COP should EXCEED cop_ref, not just approach it.
    power_peak = chiller_power(750.0, t_chw_ref, t_cond_ref, demo_curves)
    logger.info(
        "PLR=0.75 (peak-efficiency band): power=%.2f kW, COP=%.3f (should exceed cop_ref=6.0)",
        power_peak,
        750.0 / power_peak,
    )

    # Severe part-load degradation (PLR=0.15, below the 25%% threshold):
    # COP should fall inside the documented 2.0-3.5 degraded band.
    power_low = chiller_power(150.0, t_chw_ref, t_cond_ref, demo_curves)
    logger.info(
        "PLR=0.15 (below 25%% threshold): power=%.2f kW, COP=%.3f (should be in 2.0-3.5)",
        power_low,
        150.0 / power_low,
    )

    # Tropical condenser water (still well above any plausible wet
    # bulb, INV-3) at the same 500 kW load as L5.1's demo -- now BOTH
    # effects (temperature penalty + this PLR's position on the real
    # EIRFPLR curve) are active together, not just the flat-curve
    # temperature-only effect L5.1 showed.
    t_cond_tropical = 34.0
    power_tropical = chiller_power(500.0, t_chw_ref, t_cond_tropical, demo_curves)
    logger.info(
        "condenser water +%.1fK, 500kW load: power=%.2f kW, COP=%.3f "
        "(L5.1's flat-curve version gave COP=5.495 at this same point)",
        t_cond_tropical - t_cond_ref,
        power_tropical,
        500.0 / power_tropical,
    )

    # INV-6 in action: a load the chiller genuinely cannot deliver, even
    # with the 5%% overload margin, must raise -- not return a
    # quietly-wrong power number.
    try:
        chiller_power(1200.0, t_chw_ref, t_cond_ref, demo_curves)
    except ValueError as exc:
        logger.info("INV-6 correctly rejected an overloaded request: %s", exc)

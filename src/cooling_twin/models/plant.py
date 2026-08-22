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

M8 adds the second half of the plant: `plant_electric_kw()` turns a
series of COOLING LOADS (kW thermal -- what the twin predicts and what
BDG2's `chilledwater` meter measures) into ELECTRIC POWER (kW -- what a
counterfactual is answered in). That conversion is where the chiller
and the pump disagree with each other, and quantifying the disagreement
is what L8.2 exists to do.

The conversion is UNCALIBRATED and cannot be calibrated from this
dataset: BDG2 records no chiller sub-meter. See `config/plant.yaml` for
the full statement of what that costs a claim.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import yaml

from cooling_twin.models.chiller import (
    ChillerCurves,
    CurveCoeffs3,
    CurveCoeffs6,
    biquadratic,
    chiller_power,
)
from cooling_twin.models.pump import PumpParams, pump_power
from cooling_twin.models.tower import cooling_tower_outlet

logger = logging.getLogger(__name__)

# Public because the ventilation term in models/rc.py (ADR-011) must use
# the SAME latent heat of vaporisation this coil model uses -- two values
# for h_fg in one repository would put a silent inconsistency between the
# plant model and the load model that calibrates against it.
H_FG_J_PER_KG = 2_450_000.0  # 03_DOMAIN_REFERENCE.md SS3: h_fg ~ 2450 kJ/kg near 25degC

_CP_AIR_J_PER_KG_K = 1006.0  # standard dry air specific heat -- physical constant
_H_FG_J_PER_KG = H_FG_J_PER_KG  # retained name for this module's own use

# Water properties near chilled-water temperatures. Physical constants,
# not tuning knobs: cp = 4.186 kJ/kgK and rho = 1000 kg/m3 at ~7 degC.
WATER_CP_J_PER_KG_K = 4186.0
WATER_DENSITY_KG_PER_M3 = 1000.0

DEFAULT_PLANT_CONFIG_PATH = Path("config/plant.yaml")

# Per-unit load ceiling used when deciding how many chillers to run.
# INV-6 allows PLR up to 1.05 and INV-4 allows load up to 1.1x
# nameplate; staging to the TIGHTER of the two means the sequencing
# logic can never hand `chiller_power()` a request that its own
# invariant checks reject. Staging to `q_avail` alone would do exactly
# that on cold hours, where CAPFT rises above 1.1.
_STAGING_PLR_CEILING = 1.05

# INV-1 (03_DOMAIN_REFERENCE.md SS2): 0 < COP < 10. Duplicated from
# chiller.py's `_MAX_COP` rather than imported because that name is
# private there and enforces the invariant on the RATING POINT; this one
# enforces it on the hourly operating point, which is a different check
# on the same number.
_MAX_COP_INV1 = 10.0

_WATTS_PER_KW = 1000.0


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


@dataclass(frozen=True)
class PlantParams:
    """One chilled-water plant, as a set of stated assumptions.

    Frozen for the same reason as `ChillerCurves` (L5.1) and `RCParams`
    (L4.2): a counterfactual builds a NEW plant per scenario rather than
    mutating one, so that the baseline arm cannot be modified by the
    intervened arm halfway through a comparison.

    Every field is an assumption from `config/plant.yaml` or a value
    derived from the meter, never a calibrated parameter -- BDG2 has no
    chiller sub-meter to calibrate against.

    Attributes:
        curves: Performance curves and rating point of ONE chiller.
        n_chillers: Identical units in the plant, ideally sequenced.
        t_chw_supply_c: Chilled-water supply setpoint, degC.
        design_delta_t_k: Design chilled-water delta-T, K. Sets the
            flow the pump is rated for.
        tower_approach_k: Condenser water = wet bulb + this (INV-3).
        chw_pump: Rated flow and power of the chilled-water pump.
        variable_speed: True for a VFD pump (affinity law), False for
            fixed speed with valve throttling.
        t_chw_supply_limits_c: `(min, max)` the CHW supply temperature
            is clamped to before the curves are evaluated.
        t_cond_water_limits_c: `(min, max)` for the condenser water
            temperature, same purpose. See `config/plant.yaml`.
    """

    curves: ChillerCurves
    n_chillers: int
    t_chw_supply_c: float
    design_delta_t_k: float
    tower_approach_k: float
    chw_pump: PumpParams
    variable_speed: bool
    t_chw_supply_limits_c: tuple[float, float] = (4.0, 12.0)
    t_cond_water_limits_c: tuple[float, float] = (20.0, 40.0)

    def __post_init__(self) -> None:
        if self.n_chillers < 1:
            raise ValueError(f"n_chillers must be >= 1, got {self.n_chillers}")
        for name, (low, high) in (
            ("t_chw_supply_limits_c", self.t_chw_supply_limits_c),
            ("t_cond_water_limits_c", self.t_cond_water_limits_c),
        ):
            if not low < high:
                raise ValueError(f"{name}: lower limit {low} must be below upper {high}")
        if self.design_delta_t_k <= 0:
            raise ValueError(
                f"design_delta_t_k must be > 0, got {self.design_delta_t_k}. A "
                "zero or negative design delta-T violates INV-2 and would "
                "divide the flow calculation by zero."
            )

    @property
    def capacity_kw(self) -> float:
        """Total nameplate cooling capacity of the plant, kW."""
        return self.n_chillers * self.curves.q_ref_kw


@dataclass(frozen=True)
class PlantOperation:
    """One hourly series of plant operation, as electric power.

    Returned as separate chiller and pump series rather than one total,
    because the whole point of L8.2 is that an intervention can move
    them in OPPOSITE directions. A single `total_kw` series would hide
    the trade-off the lesson exists to quantify.

    Attributes:
        chiller_kw: Compressor electric power at each hour, kW.
        pump_kw: Chilled-water pump electric power at each hour, kW.
        plr: Part-load ratio of each RUNNING chiller (0.0 when off).
        n_online: Chillers running at each hour.
        flow_m3_per_s: Chilled-water flow at each hour.
        t_cond_water_c: Condenser water temperature at each hour.
        n_hours_flow_capped: Hours where the required flow exceeded the
            pump's rated flow and was capped at it. Non-zero means the
            scenario is not physically deliverable by this plant and its
            pump penalty is UNDERSTATED.
        n_hours_capacity_short: Hours where the load exceeded the
            plant's staged capacity and was capped. Non-zero means the
            plant is undersized for the scenario.
        n_hours_cop_capped: Hours where the curve set produced a COP
            above INV-1's ceiling and the power was raised to hold the
            invariant. Non-zero means the curves are being asked about
            conditions they do not describe -- report the count, never
            hide it.
    """

    chiller_kw: npt.NDArray[np.float64]
    pump_kw: npt.NDArray[np.float64]
    plr: npt.NDArray[np.float64]
    n_online: npt.NDArray[np.int_]
    flow_m3_per_s: npt.NDArray[np.float64]
    t_cond_water_c: npt.NDArray[np.float64]
    n_hours_flow_capped: int
    n_hours_capacity_short: int
    n_hours_cop_capped: int = 0

    @property
    def total_kw(self) -> npt.NDArray[np.float64]:
        """Chiller + pump electric power at each hour, kW."""
        return self.chiller_kw + self.pump_kw

    @property
    def annual_mwh(self) -> dict[str, float]:
        """Energy over the series, MWh, assuming hourly samples."""
        return {
            "chiller": float(self.chiller_kw.sum()) / _WATTS_PER_KW,
            "pump": float(self.pump_kw.sum()) / _WATTS_PER_KW,
            "total": float(self.total_kw.sum()) / _WATTS_PER_KW,
        }


def chilled_water_flow_m3_per_s(
    load_kw: npt.ArrayLike, delta_t_k: float
) -> npt.NDArray[np.float64]:
    """Flow needed to carry a cooling load at a given water delta-T.

        Q = m_dot * cp * deltaT  ->  V_dot = Q / (rho * cp * deltaT)

    This one line is where low-delta-T syndrome enters the project.
    Delta-T sits in the DENOMINATOR, so a coil that returns water 2 K
    colder than design does not cost 2 K of anything -- it costs the
    flow ratio, which the affinity law then cubes. L1.4's "2x flow = 8x
    power" is this equation followed by `pump_power()`.

    Args:
        load_kw: Cooling load carried by the loop, kW. Must be >= 0.
        delta_t_k: Supply-to-return temperature difference, K. Must be
            > 0 (INV-2: supply is strictly colder than return).

    Returns:
        Volumetric flow at each entry, m^3/s.

    Raises:
        ValueError: If `delta_t_k <= 0`, or if any load is negative or
            non-finite.
    """
    if delta_t_k <= 0:
        raise ValueError(
            f"delta_t_k must be > 0, got {delta_t_k}. INV-2 requires supply "
            "strictly colder than return; a non-positive delta-T means the "
            "loop is carrying heat the wrong way."
        )
    load = np.asarray(load_kw, dtype=float)
    if not np.all(np.isfinite(load)):
        raise ValueError("load_kw must be finite")
    if np.any(load < 0.0):
        raise ValueError(
            "load_kw must be >= 0. A negative cooling load reaching the flow "
            "calculation means the inverse model's clip at zero was skipped."
        )
    return load * _WATTS_PER_KW / (WATER_DENSITY_KG_PER_M3 * WATER_CP_J_PER_KG_K * delta_t_k)


def plant_electric_kw(
    load_kw: npt.ArrayLike,
    t_wet_bulb_c: npt.ArrayLike,
    params: PlantParams,
    t_chw_supply_c: float | None = None,
    chw_delta_t_k: float | None = None,
) -> PlantOperation:
    """Convert an hourly cooling load into chiller + pump electric power.

    Three physical steps, in the order a plant actually experiences
    them:

    1. **Heat rejection.** The tower can only reach wet bulb plus its
       approach (INV-3), so the condenser temperature -- and with it the
       chiller's efficiency -- is set by the weather, hour by hour.
    2. **Staging.** The fewest units that can carry the load run, and
       share it equally. Without this the plant would spend most of the
       year below PLR 0.25, where L5.2's EIRFPLR curve collapses, and
       every M8 result would be a statement about running one oversized
       machine rather than about the building.
    3. **Distribution.** The load and the delta-T set the flow; the
       affinity law turns the flow into pump power.

    `t_chw_supply_c` and `chw_delta_t_k` are separate arguments rather
    than being read from `params` because an INTERVENTION changes one
    without changing the other: raising the supply temperature while the
    coils hold their return temperature narrows delta-T, and that
    narrowing is the whole trade-off. Forcing the caller to state both
    makes it impossible to change one and silently keep the other
    consistent by accident.

    Args:
        load_kw: Cooling load at each hour, kW thermal. Must be >= 0.
        t_wet_bulb_c: Outdoor wet bulb at each hour, degC.
        params: The plant.
        t_chw_supply_c: Chilled-water supply temperature for this run.
            Defaults to the plant's own setpoint.
        chw_delta_t_k: Chilled-water delta-T for this run. Defaults to
            the plant's design delta-T.

    Returns:
        A `PlantOperation`.

    Raises:
        ValueError: If the two series differ in length, are empty, or if
            any input is non-finite; also for any violation raised by
            `chiller_power()` (INV-1, INV-4, INV-6) or
            `chilled_water_flow_m3_per_s()` (INV-2).
    """
    load = np.asarray(load_kw, dtype=float)
    wet_bulb = np.asarray(t_wet_bulb_c, dtype=float)
    if load.shape != wet_bulb.shape:
        raise ValueError(
            f"load_kw {load.shape} and t_wet_bulb_c {wet_bulb.shape} must have "
            "the same shape -- a mismatch here silently scores the plant "
            "against the wrong weather."
        )
    if load.size == 0:
        raise ValueError("load_kw must contain at least one hour")
    if not (np.all(np.isfinite(load)) and np.all(np.isfinite(wet_bulb))):
        raise ValueError("load_kw and t_wet_bulb_c must be finite")

    supply_c = params.t_chw_supply_c if t_chw_supply_c is None else float(t_chw_supply_c)
    delta_t_k = params.design_delta_t_k if chw_delta_t_k is None else float(chw_delta_t_k)

    t_cond = np.array(
        [cooling_tower_outlet(float(value), params.tower_approach_k) for value in wet_bulb]
    )
    # The PHYSICAL condenser temperature is reported; a CLAMPED copy is
    # what the curves are evaluated at. Keeping both is deliberate --
    # collapsing them would quietly rewrite the weather to suit the
    # curve fit, and the reported tower outlet would stop being the
    # tower outlet.
    curve_supply_c = float(np.clip(supply_c, *params.t_chw_supply_limits_c))
    curve_cond_c = np.clip(t_cond, *params.t_cond_water_limits_c)

    q_ref = params.curves.q_ref_kw
    chiller_kw = np.zeros_like(load)
    plr = np.zeros_like(load)
    n_online = np.zeros(load.shape, dtype=int)
    capacity_short = 0
    cop_capped = 0

    for index, (hour_load, t_curve_c) in enumerate(zip(load, curve_cond_c, strict=True)):
        if hour_load <= 0.0:
            continue
        capft = biquadratic(curve_supply_c, float(t_curve_c), params.curves.cap_ft)
        if capft <= 0.0:
            raise ValueError(
                f"CAPFT evaluated to {capft:.4f} at CHW {curve_supply_c} degC / "
                f"condenser {t_curve_c:.1f} degC -- the curve is being "
                "extrapolated outside the range it was fitted for."
            )
        q_avail = q_ref * capft
        # Stage to the tighter of "what the curve says is available" and
        # "what INV-4/INV-6 allow a single unit to be asked for".
        per_unit_ceiling = min(q_avail, _STAGING_PLR_CEILING * q_ref)
        needed = math.ceil(hour_load / per_unit_ceiling)
        online = min(needed, params.n_chillers)
        served = min(hour_load, online * per_unit_ceiling)
        if served < hour_load:
            capacity_short += 1
        unit_load = served / online
        power = online * chiller_power(unit_load, curve_supply_c, float(t_curve_c), params.curves)
        # INV-1 (0 < COP < 10) on the OUTPUT, not just on `cop_ref`.
        # `ChillerCurves.__post_init__` checks the rating point; nothing
        # until here checks what the three curves multiply out to at an
        # operating point far from it. Capping rather than raising is a
        # judgement: a raise would make a whole year unrunnable because
        # of a handful of cold hours, and the count is carried in the
        # result and into the artifact so the compromise is visible.
        if power > 0.0 and served / power >= _MAX_COP_INV1:
            power = served / _MAX_COP_INV1
            cop_capped += 1
        chiller_kw[index] = power
        plr[index] = unit_load / q_avail
        n_online[index] = online

    if cop_capped:
        logger.warning(
            "%d hour(s) produced COP >= %.1f and were capped at it (INV-1). "
            "That is the curve set extrapolating, not the plant being "
            "brilliant -- see config/plant.yaml's curve limits.",
            cop_capped,
            _MAX_COP_INV1,
        )
    if capacity_short:
        logger.warning(
            "%d hour(s) exceeded the plant's staged capacity and were capped. "
            "The plant is undersized for this scenario and its energy is "
            "UNDERSTATED on those hours.",
            capacity_short,
        )

    flow = chilled_water_flow_m3_per_s(load, delta_t_k)
    rated_flow = params.chw_pump.flow_rated_m3_per_s
    flow_capped = int(np.count_nonzero(flow > rated_flow))
    if flow_capped:
        logger.warning(
            "%d hour(s) required more than the pump's rated flow (%.3f m3/s) "
            "and were capped at it. The pump cannot exceed rated speed, so "
            "this scenario is NOT deliverable as modelled and its pump "
            "penalty is understated.",
            flow_capped,
            rated_flow,
        )
        flow = np.minimum(flow, rated_flow)

    pump_kw = np.array(
        [pump_power(float(value), params.chw_pump, params.variable_speed) for value in flow]
    )

    return PlantOperation(
        chiller_kw=chiller_kw,
        pump_kw=pump_kw,
        plr=plr,
        n_online=n_online,
        flow_m3_per_s=flow,
        t_cond_water_c=t_cond,
        n_hours_flow_capped=flow_capped,
        n_hours_capacity_short=capacity_short,
        n_hours_cop_capped=cop_capped,
    )


def load_plant_config(path: Path = DEFAULT_PLANT_CONFIG_PATH) -> dict:
    """Load the plant assumptions from config/plant.yaml.

    Args:
        path: Path to the plant config YAML.

    Returns:
        The parsed config as a nested dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it does not parse to a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Plant config not found at {path}. Every plant assumption is "
            "stated in that file precisely because none of them can be "
            "calibrated from BDG2 -- hardcoding one here would hide it."
        )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"{path} must parse to a YAML mapping, got {type(config).__name__}")
    return config


def build_plant_params(peak_load_kw: float, config: dict) -> PlantParams:
    """Size a plant for one building from its measured peak load.

    The plant is sized FROM THE METER rather than typed in. A nameplate
    capacity chosen by hand decides every part-load ratio in M8, and a
    part-load ratio decides the chiller's efficiency -- so a hand-picked
    nameplate is a hand-picked answer. Sizing from the largest hour the
    building actually drew makes the same rule apply to every building,
    including the ones it does not flatter.

    The pump is sized RELATIVE to the chiller using the documented HVAC
    energy split (03_DOMAIN_REFERENCE.md SS1). That split is an annual
    ENERGY share being used here as a design POWER ratio, which assumes
    equal load factors -- see `config/plant.yaml` for why that is the
    weakest assumption in the file and how L8.2 reports around it.

    Args:
        peak_load_kw: Largest measured hourly cooling load, kW.
        config: Parsed `config/plant.yaml`.

    Returns:
        A `PlantParams` sized for this building.

    Raises:
        ValueError: If `peak_load_kw` is not positive, or if a required
            config key is missing.
    """
    if peak_load_kw <= 0:
        raise ValueError(f"peak_load_kw must be > 0, got {peak_load_kw}")

    try:
        chiller_config = config["chiller"]
        sizing = config["sizing"]
        water = config["water_loop"]
        pump_config = config["pump"]
    except KeyError as error:
        raise ValueError(f"config/plant.yaml is missing section {error}") from error

    n_chillers = int(sizing["n_chillers"])
    capacity_kw = float(sizing["capacity_margin"]) * float(peak_load_kw)
    q_ref_kw = capacity_kw / n_chillers

    cap_ft: CurveCoeffs6 = tuple(float(value) for value in chiller_config["cap_ft"])  # type: ignore[assignment]
    eir_ft: CurveCoeffs6 = tuple(float(value) for value in chiller_config["eir_ft"])  # type: ignore[assignment]
    eir_fplr: CurveCoeffs3 = tuple(float(value) for value in chiller_config["eir_fplr"])  # type: ignore[assignment]
    curves = ChillerCurves(
        cap_ft=cap_ft,
        eir_ft=eir_ft,
        eir_fplr=eir_fplr,
        q_ref_kw=q_ref_kw,
        cop_ref=float(chiller_config["cop_ref"]),
    )

    design_delta_t_k = float(water["design_delta_t_k"])
    t_chw_supply_c = float(water["t_chw_supply_c"])

    # Design point: the whole plant at nameplate, at the rating
    # temperatures the curves were normalised at.
    design_chiller_kw = n_chillers * chiller_power(
        q_ref_kw,
        float(chiller_config["t_chw_supply_ref_c"]),
        float(chiller_config["t_cond_water_ref_c"]),
        curves,
    )
    pump_share = float(pump_config["share_of_hvac_pct"]) / float(
        pump_config["chiller_share_of_hvac_pct"]
    )
    chw_pump = PumpParams(
        flow_rated_m3_per_s=float(
            chilled_water_flow_m3_per_s([capacity_kw], design_delta_t_k)[0]
        ),
        power_rated_kw=pump_share * design_chiller_kw,
    )

    logger.info(
        "plant sized from a %.0f kW peak: %d x %.0f kW chillers (%.0f kW total), "
        "CHW pump %.0f kW at %.3f m3/s",
        peak_load_kw,
        n_chillers,
        q_ref_kw,
        capacity_kw,
        chw_pump.power_rated_kw,
        chw_pump.flow_rated_m3_per_s,
    )
    return PlantParams(
        curves=curves,
        n_chillers=n_chillers,
        t_chw_supply_c=t_chw_supply_c,
        design_delta_t_k=design_delta_t_k,
        tower_approach_k=float(water["tower_approach_k"]),
        chw_pump=chw_pump,
        variable_speed=bool(pump_config["variable_speed"]),
        t_chw_supply_limits_c=(
            float(chiller_config["t_chw_supply_limits_c"][0]),
            float(chiller_config["t_chw_supply_limits_c"][1]),
        ),
        t_cond_water_limits_c=(
            float(chiller_config["t_cond_water_limits_c"][0]),
            float(chiller_config["t_cond_water_limits_c"][1]),
        ),
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

    # --- M8: cooling load -> electric power, and the trade-off ---------
    #
    # A synthetic year (a sinusoidal load with a matching wet bulb) so
    # this demo runs with no BDG2 download. The real run is
    # scripts/run_counterfactual.py.
    hours = np.arange(8760, dtype=float)
    season = np.sin(2.0 * np.pi * (hours - 2400.0) / 8760.0)
    demo_load_kw = 4000.0 + 3000.0 * np.clip(season, -0.5, None)
    demo_wet_bulb_c = 12.0 + 10.0 * season

    plant_config = load_plant_config()
    plant = build_plant_params(float(demo_load_kw.max()), plant_config)

    baseline = plant_electric_kw(demo_load_kw, demo_wet_bulb_c, plant)
    logger.info(
        "baseline: chiller %.0f MWh, pump %.0f MWh, total %.0f MWh; mean plant COP %.2f",
        baseline.annual_mwh["chiller"],
        baseline.annual_mwh["pump"],
        baseline.annual_mwh["total"],
        demo_load_kw.sum() / baseline.total_kw.sum(),
    )

    # THE TRADE-OFF (L1.4, quantified). Raising chilled-water supply by
    # 2 K makes the chiller more efficient. What it does to the pump
    # depends entirely on whether the coils can still return water at
    # the same temperature:
    #   - coils rebalanced: delta-T unchanged, flow unchanged, pure win
    #   - return fixed:     delta-T narrows 6 K -> 4 K, flow x1.5,
    #                       pump power x1.5^3 = x3.375
    for label, delta_t_k in (
        ("coils hold delta-T", plant.design_delta_t_k),
        ("return fixed (low delta-T)", plant.design_delta_t_k - 2.0),
    ):
        scenario = plant_electric_kw(
            demo_load_kw,
            demo_wet_bulb_c,
            plant,
            t_chw_supply_c=plant.t_chw_supply_c + 2.0,
            chw_delta_t_k=delta_t_k,
        )
        logger.info(
            "CHW supply +2 K, %-26s chiller %+6.1f%%, pump %+8.1f%%, total %+6.1f%%",
            label + ":",
            100.0
            * (scenario.annual_mwh["chiller"] / baseline.annual_mwh["chiller"] - 1.0),
            100.0 * (scenario.annual_mwh["pump"] / baseline.annual_mwh["pump"] - 1.0),
            100.0 * (scenario.annual_mwh["total"] / baseline.annual_mwh["total"] - 1.0),
        )

"""What-if simulation -- the question a regression cannot answer.

A calibrated model that only reproduces history is a curve fit with
extra steps. The reason to build a physical twin instead of a regression
is that you can INTERVENE on it: hold the weather, the occupancy and the
building fabric exactly as they were, change one input to a value it
never took, and re-solve.

The distinction M8 rests on:

    OBSERVATIONAL   P(load | setpoint = 24)
        "on the hours when the setpoint happened to be 24, what was the
         load" -- answerable from data, and contaminated by every reason
         the setpoint happened to be 24 on those hours.

    INTERVENTIONAL  P(load | do(setpoint = 24))
        "if the setpoint were SET to 24 on every hour, holding the rest
         of the world fixed, what would the load be" -- not answerable
         from this data at all, because the setpoint is not recorded and
         never varies in the record. The twin answers it structurally.

WHAT THIS MODULE CANNOT DO, stated here because it is the first thing a
reviewer should ask. `t_setpoint_c` is a CALIBRATED PARAMETER, not a
measured thermostat setting -- BDG2 records no setpoint. It was fitted
alongside four other parameters and it absorbs whatever else in the
building behaves like a temperature offset. `do(t_setpoint_c + 1)` is
therefore an intervention on the MODEL's setpoint parameter. Whether the
building's thermostat behaves the same way is an assumption, and one
this dataset cannot test. Every number this module produces inherits
that caveat; `reports/08_counterfactual.md` states it beside the
results rather than in a footnote.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from cooling_twin.models.plant import PlantOperation, PlantParams, plant_electric_kw
from cooling_twin.models.rc import DEFAULT_SUPPLY_HUMIDITY_RATIO, inverse_cooling_load

logger = logging.getLogger(__name__)

_PERCENT = 100.0
_KW_TO_MW = 1000.0

# Parameter names the twin needs. Stated as a constant so a missing one
# fails with a readable message instead of a KeyError from inside the
# ODE call.
REQUIRED_PARAMETERS = (
    "ua_envelope_w_per_m2k",
    "r_internal_ratio",
    "internal_gain_w_per_m2",
    "t_setpoint_c",
)


@dataclass(frozen=True)
class Scenario:
    """One intervention, stated as deltas from the calibrated baseline.

    Deltas rather than absolute values, deliberately. An absolute
    setpoint of 23.0 degC means something different for every building
    (each has its own fitted `t_setpoint_c`, and Claude's is 24.8 degC
    -- a number no thermostat was ever set to), so a scenario written in
    absolutes silently becomes a different intervention on each
    building. `+1 K` is the same intervention everywhere.

    Attributes:
        name: Short identifier, used as the artifact key.
        description: What an operator would actually do.
        zone_setpoint_delta_c: Change to the zone air setpoint, K.
        chw_supply_delta_c: Change to the chilled-water supply
            temperature, K. Affects the plant only -- the zone still
            gets the cooling it needs, delivered by colder or warmer
            water.
        chw_return_fixed: When the supply temperature moves, does the
            RETURN temperature follow it? `False` assumes the coils are
            rebalanced so delta-T is preserved (flow unchanged). `True`
            assumes the return stays where it was, so delta-T narrows by
            `chw_supply_delta_c` and the flow -- and cubically, the pump
            power -- rises. This single boolean is the difference
            between the intervention saving energy and costing it.
        vent_flow_scale: Multiplier on the calibrated ventilation flow.
            0.7 is a 30% outside-air setback.
    """

    name: str
    description: str
    zone_setpoint_delta_c: float = 0.0
    chw_supply_delta_c: float = 0.0
    chw_return_fixed: bool = False
    vent_flow_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.vent_flow_scale < 0.0:
            raise ValueError(
                f"vent_flow_scale must be >= 0, got {self.vent_flow_scale}. A "
                "negative outside-air flow would supply cooling from nowhere."
            )
        if not self.name:
            raise ValueError("scenario name must not be empty")

    @property
    def touches_load(self) -> bool:
        """Whether this scenario changes the building's cooling load.

        A chilled-water scenario changes how the load is SERVED, not how
        big it is. Re-solving the ODE for it would return the identical
        series at the cost of a second simulation -- and, worse, would
        invite the reader to believe the zone model had something to say
        about a plant-side change.
        """
        return self.zone_setpoint_delta_c != 0.0 or self.vent_flow_scale != 1.0


# The three setpoint scenarios 06_ASSESSMENT.md's M8 gate asks for, plus
# the two that turn the chiller-versus-pump trade-off from an assertion
# into a number. The pair differ ONLY in `chw_return_fixed`.
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="zone_setpoint_plus_1k",
        description="Relax the zone air setpoint by 1 K (e.g. 24 -> 25 degC).",
        zone_setpoint_delta_c=1.0,
    ),
    Scenario(
        name="zone_setpoint_minus_1k",
        description="Tighten the zone air setpoint by 1 K -- the cost of a comfort complaint.",
        zone_setpoint_delta_c=-1.0,
    ),
    Scenario(
        name="chw_supply_plus_2k_coils_hold",
        description=(
            "Raise chilled-water supply by 2 K, coils rebalanced so delta-T "
            "(and therefore flow) is unchanged."
        ),
        chw_supply_delta_c=2.0,
        chw_return_fixed=False,
    ),
    Scenario(
        name="chw_supply_plus_2k_return_fixed",
        description=(
            "Raise chilled-water supply by 2 K with the return temperature "
            "unchanged -- delta-T narrows, flow rises, pump power cubes."
        ),
        chw_supply_delta_c=2.0,
        chw_return_fixed=True,
    ),
    Scenario(
        name="ventilation_setback_30pct",
        description="Cut calibrated outside-air flow by 30% during the year.",
        vent_flow_scale=0.7,
    ),
)


@dataclass(frozen=True)
class CalibratedTwin:
    """The calibrated model plus the drivers it runs on, ready to intervene on.

    Holds the FROZEN calibrated parameters and one building-year of
    weather. Frozen, and the parameters are copied in rather than
    referenced, because a counterfactual that could mutate the baseline
    mid-comparison is not a comparison.

    Attributes:
        t_seconds: Seconds since the first sample, strictly increasing.
        t_ambient_c: Outdoor dry bulb at each hour, degC.
        humidity_ratio: Outdoor humidity ratio at each hour, kg/kg.
        wet_bulb_c: Outdoor wet bulb at each hour, degC. Only the plant
            uses it (the tower is wet-bulb limited, INV-3).
        floor_area_m2: Conditioned floor area.
        parameters: The calibrated parameter set, by name.
        envelope_capacity_ratio: Fixed at nominal by L6.5's screening.
        ceiling_height_m: Same.
        supply_humidity_ratio: ADR-011's stated coil assumption.
        building_id: For logs and artifacts.
        year: The year the drivers come from.
    """

    t_seconds: npt.NDArray[np.float64]
    t_ambient_c: npt.NDArray[np.float64]
    humidity_ratio: npt.NDArray[np.float64]
    wet_bulb_c: npt.NDArray[np.float64]
    floor_area_m2: float
    parameters: Mapping[str, float]
    envelope_capacity_ratio: float = 20.0
    ceiling_height_m: float = 3.0
    supply_humidity_ratio: float = DEFAULT_SUPPLY_HUMIDITY_RATIO
    building_id: str = "unknown"
    year: int = 0

    def __post_init__(self) -> None:
        missing = [name for name in REQUIRED_PARAMETERS if name not in self.parameters]
        if missing:
            raise ValueError(
                f"calibrated parameters missing {missing}. The twin runs the "
                "same inverse model the calibration scored, so it needs the "
                "same parameter set -- not a subset."
            )
        lengths = {
            len(self.t_seconds),
            len(self.t_ambient_c),
            len(self.humidity_ratio),
            len(self.wet_bulb_c),
        }
        if len(lengths) != 1:
            raise ValueError(
                f"driver series have mismatched lengths {sorted(lengths)}; a "
                "counterfactual scored against the wrong hours is worse than "
                "no counterfactual."
            )

    def predict_load_kw(
        self, setpoint_delta_c: float = 0.0, vent_flow_scale: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Cooling load under an intervention on the model's inputs.

        This is the `do()` operator, concretely: the drivers are held
        exactly as they were, one parameter is overwritten, and the ODE
        is re-solved from the same initial condition. Nothing is
        re-fitted. If anything were re-fitted the answer would be
        "what parameter set best explains a world where the setpoint was
        different", which is a different -- and unanswerable -- question.

        Args:
            setpoint_delta_c: Added to the calibrated `t_setpoint_c`.
            vent_flow_scale: Multiplies the calibrated ventilation flow.

        Returns:
            Required cooling at each hour, kW, clipped at zero.

        Raises:
            RuntimeError: If the ODE solver fails.
            ValueError: For any invalid parameter combination, as raised
                by `inverse_cooling_load`.
        """
        vent_flow = float(self.parameters.get("vent_flow_kg_per_s", 0.0)) * vent_flow_scale
        clipped_kw, _raw_kw = inverse_cooling_load(
            self.t_seconds,
            self.t_ambient_c,
            ua_envelope_w_per_m2k=float(self.parameters["ua_envelope_w_per_m2k"]),
            r_internal_ratio=float(self.parameters["r_internal_ratio"]),
            internal_gain_w_per_m2=float(self.parameters["internal_gain_w_per_m2"]),
            t_setpoint_c=float(self.parameters["t_setpoint_c"]) + setpoint_delta_c,
            floor_area_m2=self.floor_area_m2,
            envelope_capacity_ratio=self.envelope_capacity_ratio,
            ceiling_height_m=self.ceiling_height_m,
            vent_flow_kg_per_s=vent_flow,
            outdoor_humidity_ratio=(None if vent_flow <= 0.0 else self.humidity_ratio),
            supply_humidity_ratio=self.supply_humidity_ratio,
        )
        return clipped_kw


@dataclass(frozen=True)
class CounterfactualResult:
    """One scenario, scored against the baseline it was compared to.

    Both arms are kept whole rather than reduced to a percentage. The
    percentage is what gets quoted; the series is what makes the
    percentage checkable, and the plant split is what shows whether the
    chiller and the pump moved together or against each other.

    Attributes:
        scenario: The intervention.
        baseline_load_kw: Cooling load, calibrated baseline.
        scenario_load_kw: Cooling load under the intervention.
        baseline_plant: Plant operation on the baseline load.
        scenario_plant: Plant operation under the intervention.
        notes: Free-form flags raised while running (capacity shortfall,
            capped flow, capped COP).
    """

    scenario: Scenario
    baseline_load_kw: npt.NDArray[np.float64]
    scenario_load_kw: npt.NDArray[np.float64]
    baseline_plant: PlantOperation
    scenario_plant: PlantOperation
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def load_change_pct(self) -> float:
        """Change in annual cooling load (thermal), percent."""
        return _relative_change_pct(self.baseline_load_kw.sum(), self.scenario_load_kw.sum())

    @property
    def chiller_change_pct(self) -> float:
        """Change in annual chiller electricity, percent."""
        return _relative_change_pct(
            self.baseline_plant.chiller_kw.sum(), self.scenario_plant.chiller_kw.sum()
        )

    @property
    def pump_change_pct(self) -> float:
        """Change in annual pump electricity, percent."""
        return _relative_change_pct(
            self.baseline_plant.pump_kw.sum(), self.scenario_plant.pump_kw.sum()
        )

    @property
    def total_change_pct(self) -> float:
        """Change in annual plant electricity, percent."""
        return _relative_change_pct(
            self.baseline_plant.total_kw.sum(), self.scenario_plant.total_kw.sum()
        )

    @property
    def delta_mwh(self) -> dict[str, float]:
        """Scenario minus baseline energy, MWh. Negative is a saving."""
        base = self.baseline_plant.annual_mwh
        scenario = self.scenario_plant.annual_mwh
        delta = {key: scenario[key] - base[key] for key in base}
        delta["cooling_load"] = float(
            (self.scenario_load_kw.sum() - self.baseline_load_kw.sum()) / _KW_TO_MW
        )
        return delta

    @property
    def hourly_total_delta_kw(self) -> npt.NDArray[np.float64]:
        """Hourly scenario-minus-baseline plant power, kW."""
        return self.scenario_plant.total_kw - self.baseline_plant.total_kw

    @property
    def trade_off(self) -> str:
        """One line naming which component won and which lost."""
        chiller = self.chiller_change_pct
        pump = self.pump_change_pct
        direction = "together" if chiller * pump >= 0.0 else "AGAINST each other"
        return (
            f"chiller {chiller:+.1f}%, pump {pump:+.1f}% -- they move {direction}; "
            f"net {self.total_change_pct:+.1f}%"
        )

    def to_dict(self) -> dict[str, Any]:
        """A JSON-shaped summary of the scenario."""
        return {
            "name": self.scenario.name,
            "description": self.scenario.description,
            "zone_setpoint_delta_c": self.scenario.zone_setpoint_delta_c,
            "chw_supply_delta_c": self.scenario.chw_supply_delta_c,
            "chw_return_fixed": self.scenario.chw_return_fixed,
            "vent_flow_scale": self.scenario.vent_flow_scale,
            "load_change_pct": self.load_change_pct,
            "chiller_change_pct": self.chiller_change_pct,
            "pump_change_pct": self.pump_change_pct,
            "total_change_pct": self.total_change_pct,
            "baseline_mwh": self.baseline_plant.annual_mwh,
            "scenario_mwh": self.scenario_plant.annual_mwh,
            "delta_mwh": self.delta_mwh,
            "baseline_mean_load_kw": float(self.baseline_load_kw.mean()),
            "scenario_mean_load_kw": float(self.scenario_load_kw.mean()),
            "n_hours": int(self.baseline_load_kw.size),
            "notes": list(self.notes),
        }


def _relative_change_pct(baseline_total: float, scenario_total: float) -> float:
    """Percent change, with a zero baseline reported as zero rather than inf."""
    if baseline_total == 0.0:
        return 0.0
    return _PERCENT * (float(scenario_total) - float(baseline_total)) / float(baseline_total)


def _scenario_notes(
    scenario_plant: PlantOperation, baseline_plant: PlantOperation
) -> tuple[str, ...]:
    """Flags that must travel with the result rather than sit in a log."""
    notes = []
    for label, operation in (("baseline", baseline_plant), ("scenario", scenario_plant)):
        if operation.n_hours_capacity_short:
            notes.append(
                f"{label}: {operation.n_hours_capacity_short} h above plant capacity, "
                "energy understated on those hours"
            )
        if operation.n_hours_flow_capped:
            notes.append(
                f"{label}: {operation.n_hours_flow_capped} h required more than rated "
                "pump flow -- NOT deliverable as modelled, pump penalty understated"
            )
        if operation.n_hours_cop_capped:
            notes.append(
                f"{label}: {operation.n_hours_cop_capped} h capped at INV-1's COP "
                "ceiling, chiller saving understated on those hours"
            )
    return tuple(notes)


def simulate_setpoint_change(
    twin: CalibratedTwin,
    scenario: Scenario,
    plant: PlantParams,
    baseline_load_kw: npt.NDArray[np.float64] | None = None,
) -> CounterfactualResult:
    """Run one intervention against the calibrated baseline.

    Both arms use the SAME weather, the SAME plant and the SAME
    parameters except the one being intervened on. That is what makes
    the difference attributable to the intervention rather than to
    anything else -- and it is why the baseline load can be passed in
    and reused across scenarios instead of being re-simulated each time
    with a fresh chance of drifting.

    Args:
        twin: The calibrated twin and its drivers.
        scenario: The intervention.
        plant: The plant that serves the load.
        baseline_load_kw: Pre-computed baseline load, to avoid
            re-solving the ODE once per scenario. Recomputed when None.

    Returns:
        A `CounterfactualResult`.

    Raises:
        ValueError: If the scenario narrows the chilled-water delta-T to
            zero or below (INV-2), or for any invalid input raised
            downstream.
        RuntimeError: If the ODE solver fails.
    """
    baseline = twin.predict_load_kw() if baseline_load_kw is None else baseline_load_kw
    scenario_load = (
        twin.predict_load_kw(
            setpoint_delta_c=scenario.zone_setpoint_delta_c,
            vent_flow_scale=scenario.vent_flow_scale,
        )
        if scenario.touches_load
        else baseline
    )

    delta_t_k = plant.design_delta_t_k
    if scenario.chw_return_fixed:
        delta_t_k = plant.design_delta_t_k - scenario.chw_supply_delta_c
        if delta_t_k <= 0.0:
            raise ValueError(
                f"scenario {scenario.name!r} raises chilled-water supply by "
                f"{scenario.chw_supply_delta_c} K against a design delta-T of "
                f"{plant.design_delta_t_k} K with the return fixed, which leaves "
                f"delta-T at {delta_t_k} K. INV-2 requires supply strictly colder "
                "than return: the coils cannot deliver this at all."
            )

    baseline_plant = plant_electric_kw(baseline, twin.wet_bulb_c, plant)
    scenario_plant = plant_electric_kw(
        scenario_load,
        twin.wet_bulb_c,
        plant,
        t_chw_supply_c=plant.t_chw_supply_c + scenario.chw_supply_delta_c,
        chw_delta_t_k=delta_t_k,
    )

    result = CounterfactualResult(
        scenario=scenario,
        baseline_load_kw=baseline,
        scenario_load_kw=scenario_load,
        baseline_plant=baseline_plant,
        scenario_plant=scenario_plant,
        notes=_scenario_notes(scenario_plant, baseline_plant),
    )
    logger.info("%-34s %s", scenario.name, result.trade_off)
    for note in result.notes:
        logger.warning("%s: %s", scenario.name, note)
    return result


def compare_scenarios(
    twin: CalibratedTwin,
    plant: PlantParams,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
) -> list[CounterfactualResult]:
    """Run every scenario against one shared baseline.

    Args:
        twin: The calibrated twin and its drivers.
        plant: The plant that serves the load.
        scenarios: Interventions to run.

    Returns:
        One result per scenario, in the order given.
    """
    baseline = twin.predict_load_kw()
    logger.info(
        "%s %d baseline: %d h, mean cooling load %.0f kW",
        twin.building_id,
        twin.year,
        baseline.size,
        float(baseline.mean()),
    )
    return [
        simulate_setpoint_change(twin, scenario, plant, baseline_load_kw=baseline)
        for scenario in scenarios
    ]

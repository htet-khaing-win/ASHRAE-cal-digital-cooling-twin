"""Unit tests for the counterfactual engine
(src/cooling_twin/twin/counterfactual.py) and the plant-power layer it
runs on (src/cooling_twin/models/plant.py, M8 additions).

Test patterns 1, 2 and 4 of 4 (05_ENGINEERING_STANDARDS.md SS3):
known-answer tests hand-computed from the physics, property tests over
random parameter draws, and input-validation tests for every raise
branch. The physical invariants are checked on the OUTPUT, not assumed:
INV-1 (COP < 10), INV-2 (delta-T > 0), INV-3 (condenser water above wet
bulb) and INV-6 (PLR <= 1.05) each have a test here.

`simulate_setpoint_change` is exercised on a short synthetic year rather
than on BDG2, so this file runs with no dataset downloaded (05
ENGINEERING_STANDARDS: H must never be blocked).
"""

from __future__ import annotations

import numpy as np
import pytest

from cooling_twin.models.chiller import ChillerCurves
from cooling_twin.models.plant import (
    PlantParams,
    build_plant_params,
    chilled_water_flow_m3_per_s,
    load_plant_config,
    plant_electric_kw,
)
from cooling_twin.twin.counterfactual import (
    DEFAULT_SCENARIOS,
    CalibratedTwin,
    Scenario,
    compare_scenarios,
    simulate_setpoint_change,
)

HOURS = 24 * 30
CALIBRATED_PARAMETERS = {
    "ua_envelope_w_per_m2k": 1.2,
    "r_internal_ratio": 6.0,
    "internal_gain_w_per_m2": 40.0,
    "vent_flow_kg_per_s": 30.0,
    "t_setpoint_c": 23.0,
}


def make_twin(**overrides: object) -> CalibratedTwin:
    """A synthetic month of drivers, warm enough that the load never clips."""
    hours = np.arange(HOURS, dtype=float)
    fields: dict = {
        "t_seconds": hours * 3600.0,
        "t_ambient_c": 24.0 + 8.0 * np.sin(2.0 * np.pi * hours / 24.0),
        "humidity_ratio": np.full(HOURS, 0.011),
        "wet_bulb_c": 18.0 + 3.0 * np.sin(2.0 * np.pi * hours / 24.0),
        "floor_area_m2": 10_000.0,
        "parameters": dict(CALIBRATED_PARAMETERS),
        "building_id": "Synthetic_test_building",
        "year": 2016,
    }
    fields.update(overrides)
    return CalibratedTwin(**fields)  # type: ignore[arg-type]


def make_plant(peak_load_kw: float = 3000.0) -> PlantParams:
    """A plant sized from the config, so the tests exercise the real file."""
    return build_plant_params(peak_load_kw, load_plant_config())


# --- flow: the equation low-delta-T syndrome lives in ---------------------


def test_flow_known_answer() -> None:
    """Q = 4186 kW at delta-T = 1 K needs exactly 1 m3/s:
    V = Q * 1000 / (rho * cp * dT) = 4186000 / (1000 * 4186 * 1) = 1.0.
    """
    flow = chilled_water_flow_m3_per_s([4186.0], delta_t_k=1.0)
    assert flow[0] == pytest.approx(1.0)


def test_halving_delta_t_doubles_the_flow() -> None:
    """Delta-T is in the denominator -- this is why a 2 K coil problem
    is a flow problem, and (through the affinity law) a cubic power
    problem.
    """
    wide = chilled_water_flow_m3_per_s([1000.0], delta_t_k=6.0)[0]
    narrow = chilled_water_flow_m3_per_s([1000.0], delta_t_k=3.0)[0]
    assert narrow == pytest.approx(2.0 * wide)


def test_flow_rejects_non_positive_delta_t() -> None:
    """INV-2: supply strictly colder than return."""
    with pytest.raises(ValueError, match="INV-2"):
        chilled_water_flow_m3_per_s([1000.0], delta_t_k=0.0)


def test_flow_rejects_negative_load() -> None:
    with pytest.raises(ValueError, match="must be >= 0"):
        chilled_water_flow_m3_per_s([-1.0], delta_t_k=6.0)


# --- the plant ------------------------------------------------------------


def test_plant_is_off_at_zero_load() -> None:
    operation = plant_electric_kw(np.zeros(10), np.full(10, 15.0), make_plant())
    assert operation.chiller_kw.sum() == pytest.approx(0.0)
    assert operation.pump_kw.sum() == pytest.approx(0.0)
    assert operation.n_online.max() == 0


def test_plant_respects_inv3_condenser_above_wet_bulb() -> None:
    """A cooling tower cannot reach wet bulb. The approach makes this
    structural rather than checked after the fact -- so the test is that
    the structure holds for every hour, not that a warning fired.
    """
    wet_bulb = np.linspace(-5.0, 30.0, 50)
    operation = plant_electric_kw(np.full(50, 2000.0), wet_bulb, make_plant())
    assert np.all(operation.t_cond_water_c > wet_bulb)


def test_plant_respects_inv6_part_load_ratio() -> None:
    """Staging must never hand a chiller more than INV-6 allows, at any
    load from nearly nothing to the plant's full capacity.
    """
    plant = make_plant()
    load = np.linspace(1.0, plant.capacity_kw, 200)
    operation = plant_electric_kw(load, np.full(200, 20.0), plant)
    assert operation.plr.max() <= 1.05


def test_plant_respects_inv1_cop_ceiling() -> None:
    """INV-1 is enforced on the hourly operating point, not just on
    `cop_ref`. Cold weather is exactly where the demo curve set tries to
    break it, so that is where this is tested.
    """
    plant = make_plant()
    load = np.full(200, 2000.0)
    operation = plant_electric_kw(load, np.full(200, -5.0), plant)
    cop = load / operation.chiller_kw
    assert cop.max() <= 10.0 + 1e-9


def test_plant_stages_the_fewest_units_that_can_carry_the_load() -> None:
    plant = make_plant()
    per_unit = plant.curves.q_ref_kw
    operation = plant_electric_kw(
        np.array([0.5 * per_unit, 1.5 * per_unit, 2.5 * per_unit]),
        np.full(3, 29.4 - 3.5),
        plant,
    )
    assert operation.n_online.tolist() == [1, 2, 3]


def test_staging_keeps_the_chiller_off_its_worst_part_load_point() -> None:
    """The reason staging is modelled at all: one oversized machine
    would run the year below PLR 0.25, where L5.2's EIRFPLR curve
    collapses, and every M8 number would be about that choice.
    """
    plant = make_plant()
    load = np.full(50, 0.55 * plant.capacity_kw)
    staged = plant_electric_kw(load, np.full(50, 20.0), plant)
    single = plant_electric_kw(
        load,
        np.full(50, 20.0),
        PlantParams(
            curves=ChillerCurves(
                cap_ft=plant.curves.cap_ft,
                eir_ft=plant.curves.eir_ft,
                eir_fplr=plant.curves.eir_fplr,
                q_ref_kw=plant.capacity_kw,
                cop_ref=plant.curves.cop_ref,
            ),
            n_chillers=1,
            t_chw_supply_c=plant.t_chw_supply_c,
            design_delta_t_k=plant.design_delta_t_k,
            tower_approach_k=plant.tower_approach_k,
            chw_pump=plant.chw_pump,
            variable_speed=plant.variable_speed,
        ),
    )
    assert staged.chiller_kw.sum() < single.chiller_kw.sum()


def test_plant_rejects_mismatched_weather() -> None:
    with pytest.raises(ValueError, match="same shape"):
        plant_electric_kw(np.ones(10), np.ones(9), make_plant())


def test_plant_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one hour"):
        plant_electric_kw(np.array([]), np.array([]), make_plant())


def test_plant_params_rejects_zero_delta_t() -> None:
    plant = make_plant()
    with pytest.raises(ValueError, match="design_delta_t_k"):
        PlantParams(
            curves=plant.curves,
            n_chillers=1,
            t_chw_supply_c=6.7,
            design_delta_t_k=0.0,
            tower_approach_k=3.5,
            chw_pump=plant.chw_pump,
            variable_speed=True,
        )


def test_plant_params_rejects_zero_chillers() -> None:
    plant = make_plant()
    with pytest.raises(ValueError, match="n_chillers"):
        PlantParams(
            curves=plant.curves,
            n_chillers=0,
            t_chw_supply_c=6.7,
            design_delta_t_k=6.0,
            tower_approach_k=3.5,
            chw_pump=plant.chw_pump,
            variable_speed=True,
        )


def test_build_plant_params_sizes_from_the_measured_peak() -> None:
    config = load_plant_config()
    plant = build_plant_params(10_000.0, config)
    margin = float(config["sizing"]["capacity_margin"])
    assert plant.capacity_kw == pytest.approx(margin * 10_000.0)
    assert plant.chw_pump.flow_rated_m3_per_s == pytest.approx(
        chilled_water_flow_m3_per_s([plant.capacity_kw], plant.design_delta_t_k)[0]
    )


def test_build_plant_params_rejects_a_non_positive_peak() -> None:
    with pytest.raises(ValueError, match="peak_load_kw"):
        build_plant_params(0.0, load_plant_config())


def test_pump_power_follows_the_cube_of_flow() -> None:
    """L1.4's rule of thumb, end to end through the plant: doubling the
    load at a fixed delta-T doubles the flow and multiplies pump power
    by eight.
    """
    plant = make_plant(peak_load_kw=20_000.0)
    low = plant_electric_kw(np.array([2000.0]), np.array([20.0]), plant)
    high = plant_electric_kw(np.array([4000.0]), np.array([20.0]), plant)
    assert high.pump_kw[0] == pytest.approx(8.0 * low.pump_kw[0], rel=1e-6)


def test_fixed_speed_pump_does_not_get_the_cubic_saving() -> None:
    plant = make_plant(peak_load_kw=20_000.0)
    fixed = PlantParams(
        curves=plant.curves,
        n_chillers=plant.n_chillers,
        t_chw_supply_c=plant.t_chw_supply_c,
        design_delta_t_k=plant.design_delta_t_k,
        tower_approach_k=plant.tower_approach_k,
        chw_pump=plant.chw_pump,
        variable_speed=False,
    )
    low = plant_electric_kw(np.array([2000.0]), np.array([20.0]), fixed)
    high = plant_electric_kw(np.array([4000.0]), np.array([20.0]), fixed)
    assert low.pump_kw[0] == pytest.approx(high.pump_kw[0])


# --- the twin -------------------------------------------------------------


def test_twin_rejects_a_missing_calibrated_parameter() -> None:
    parameters = dict(CALIBRATED_PARAMETERS)
    del parameters["t_setpoint_c"]
    with pytest.raises(ValueError, match="missing"):
        make_twin(parameters=parameters)


def test_twin_rejects_mismatched_driver_lengths() -> None:
    with pytest.raises(ValueError, match="mismatched lengths"):
        make_twin(wet_bulb_c=np.zeros(5))


def test_raising_the_setpoint_lowers_the_load() -> None:
    """The sign of the intervention, and the one property a reviewer
    checks first. A warmer zone needs less cooling -- if this ever
    reverses, a sign has flipped in the inverse model, not in the
    building.
    """
    twin = make_twin()
    baseline = twin.predict_load_kw()
    warmer = twin.predict_load_kw(setpoint_delta_c=1.0)
    cooler = twin.predict_load_kw(setpoint_delta_c=-1.0)
    assert warmer.mean() < baseline.mean() < cooler.mean()


def test_setpoint_response_is_monotone_over_random_parameter_draws() -> None:
    """Property test, 20 draws: the load must fall monotonically as the
    setpoint rises, for any physically valid parameter set -- not just
    for the one this file happens to use.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        parameters = {
            "ua_envelope_w_per_m2k": float(rng.uniform(0.3, 3.0)),
            "r_internal_ratio": float(rng.uniform(2.0, 20.0)),
            "internal_gain_w_per_m2": float(rng.uniform(5.0, 150.0)),
            "vent_flow_kg_per_s": float(rng.uniform(0.0, 100.0)),
            "t_setpoint_c": float(rng.uniform(20.0, 26.0)),
        }
        twin = make_twin(parameters=parameters)
        means = [
            twin.predict_load_kw(setpoint_delta_c=delta).mean() for delta in (-1.0, 0.0, 1.0)
        ]
        assert means[0] >= means[1] >= means[2]


def test_ventilation_setback_cannot_raise_the_load() -> None:
    twin = make_twin()
    baseline = twin.predict_load_kw()
    setback = twin.predict_load_kw(vent_flow_scale=0.7)
    assert setback.sum() <= baseline.sum()


# --- scenarios ------------------------------------------------------------


def test_scenario_rejects_negative_ventilation_scale() -> None:
    with pytest.raises(ValueError, match="vent_flow_scale"):
        Scenario(name="bad", description="", vent_flow_scale=-0.1)


def test_scenario_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="name"):
        Scenario(name="", description="")


def test_chilled_water_scenarios_do_not_touch_the_zone_load() -> None:
    """A plant-side change moves how the load is SERVED, never how big
    it is. If this ever fails, the zone model has been wired into a
    question it cannot answer.
    """
    twin = make_twin()
    result = simulate_setpoint_change(
        twin,
        Scenario(name="chw", description="", chw_supply_delta_c=2.0),
        make_plant(),
    )
    assert result.load_change_pct == pytest.approx(0.0)
    assert np.array_equal(result.baseline_load_kw, result.scenario_load_kw)


def test_null_scenario_changes_nothing() -> None:
    """The control arm. A scenario with no intervention must return
    exactly zero on every channel -- any drift here is the comparison
    machinery, not the physics.
    """
    result = simulate_setpoint_change(
        make_twin(), Scenario(name="null", description=""), make_plant()
    )
    assert result.total_change_pct == pytest.approx(0.0)
    assert result.chiller_change_pct == pytest.approx(0.0)
    assert result.pump_change_pct == pytest.approx(0.0)


def test_return_fixed_scenario_raises_the_pump_and_lowers_the_chiller() -> None:
    """The trade-off, as a test. Raising the chilled-water supply makes
    the chiller more efficient and, with the return temperature fixed,
    narrows delta-T so the pump pays cubically.
    """
    result = simulate_setpoint_change(
        make_twin(),
        Scenario(
            name="chw_return_fixed",
            description="",
            chw_supply_delta_c=2.0,
            chw_return_fixed=True,
        ),
        make_plant(),
    )
    assert result.chiller_change_pct < 0.0
    assert result.pump_change_pct > 0.0
    assert "AGAINST each other" in result.trade_off


def test_pump_penalty_matches_the_affinity_law_exactly() -> None:
    """Known answer: delta-T 6 K -> 4 K is a flow ratio of 1.5, so pump
    power rises by 1.5^3 - 1 = 237.5%. Any other number means the
    affinity law or the flow equation has drifted.
    """
    result = simulate_setpoint_change(
        make_twin(),
        Scenario(
            name="chw_return_fixed",
            description="",
            chw_supply_delta_c=2.0,
            chw_return_fixed=True,
        ),
        make_plant(peak_load_kw=50_000.0),
    )
    assert result.pump_change_pct == pytest.approx(100.0 * (1.5**3 - 1.0), rel=1e-6)


def test_scenario_that_would_invert_delta_t_is_refused() -> None:
    """Raising supply by 6 K with a 6 K design delta-T and a fixed
    return puts supply AT the return temperature: INV-2, and a plant
    that cannot deliver the scenario at all.
    """
    with pytest.raises(ValueError, match="INV-2"):
        simulate_setpoint_change(
            make_twin(),
            Scenario(
                name="impossible",
                description="",
                chw_supply_delta_c=6.0,
                chw_return_fixed=True,
            ),
            make_plant(),
        )


def test_compare_scenarios_runs_every_default_scenario_on_one_baseline() -> None:
    results = compare_scenarios(make_twin(), make_plant())
    assert len(results) == len(DEFAULT_SCENARIOS)
    first = results[0].baseline_load_kw
    for result in results[1:]:
        assert np.array_equal(result.baseline_load_kw, first)


def test_result_delta_mwh_agrees_with_the_percentages() -> None:
    result = simulate_setpoint_change(
        make_twin(),
        Scenario(name="warmer", description="", zone_setpoint_delta_c=1.0),
        make_plant(),
    )
    baseline_mwh = result.baseline_plant.annual_mwh["total"]
    assert result.delta_mwh["total"] == pytest.approx(
        baseline_mwh * result.total_change_pct / 100.0, rel=1e-9
    )


def test_default_scenarios_include_at_least_three_setpoint_cases() -> None:
    """06_ASSESSMENT.md's M8 gate asks for >= 3 setpoint scenarios. The
    gate item is encoded here so that deleting a scenario fails a test
    rather than quietly failing the gate.
    """
    setpoint_scenarios = [
        scenario
        for scenario in DEFAULT_SCENARIOS
        if scenario.zone_setpoint_delta_c != 0.0 or scenario.chw_supply_delta_c != 0.0
    ]
    assert len(setpoint_scenarios) >= 3

"""The scaffold is importable and the fake twin behaves as the suite assumes.

Thin on purpose: M0 ships no gateway logic, so there is nothing else to
assert about `gateway` yet. What these tests DO establish is that the
90% coverage gate is measuring a real package rather than passing on an
empty one, and that `FakeTwin` -- which every later milestone's tests
depend on -- does what its docstring claims.

`FakeTwin` is exercised here rather than only implicitly through gateway
tests, because a broken test double produces *passing* security tests,
which is the worst available outcome.
"""

from __future__ import annotations

from fakes.fake_twin import (
    DEFAULT_ZONES,
    MAX_SUMMARY_POINTS,
    FakeTwin,
)

import gateway


def test_gateway_package_imports_and_declares_a_version() -> None:
    """Nothing to test yet beyond the package existing -- say so honestly."""
    assert gateway.__version__
    assert gateway.__all__ == ["__version__"]


def test_fake_twin_scopes_zones_to_a_site() -> None:
    """Cross-tenant isolation must be visible in the double, or T-06 tests lie."""
    twin = FakeTwin()
    site_a = twin.get_zones("site_a")["zones"]
    assert {zone["zone_id"] for zone in site_a} == {"zone_north", "zone_south"}
    assert twin.get_zones("site_b")["zones"][0]["zone_id"] == "zone_main"
    assert twin.get_zones("site_nonexistent")["zones"] == []


def test_fake_twin_never_returns_more_than_the_point_cap() -> None:
    """T-12: a week-long window must not become a week of raw points."""
    summary = twin_history(minutes=10080)
    assert summary["n_points"] <= MAX_SUMMARY_POINTS
    assert twin_history(minutes=60)["n_points"] == 1


def twin_history(minutes: int) -> dict[str, float]:
    """Helper: one sensor-history summary."""
    result: dict[str, float] = FakeTwin().get_sensor_history("zone_north", minutes)["summary"]
    return result


def test_identical_simulation_inputs_produce_an_identical_id() -> None:
    """M2's caching requirement, and what a receipt binds to."""
    twin = FakeTwin()
    first = twin.start_simulation("zone_north", 24.0, 24)
    second = twin.start_simulation("zone_north", 24.0, 24)
    different = twin.start_simulation("zone_north", 24.5, 24)
    assert first["simulation_id"] == second["simulation_id"]
    assert first["simulation_id"] != different["simulation_id"]
    assert first["status"] == "pending"


def test_simulation_can_be_made_to_fail_for_check_seven() -> None:
    """Receipt validation check 7 needs an error path to exercise."""
    assert FakeTwin().get_simulation_result("sim_x")["status"] == "completed"
    assert FakeTwin(fail_simulations=True).get_simulation_result("sim_x")["status"] == "error"


def test_fake_twin_accepts_out_of_range_setpoints() -> None:
    """Deliberate: it proves the GATEWAY enforced the bound, not the upstream.

    If the double refused 35 C, a passing bounds test could not tell
    which component did the refusing -- and the gateway is the control
    point precisely because the upstream may be replaced by one that
    does not check (ADR-0007).
    """
    twin = FakeTwin()
    assert twin.apply_setpoint("zone_north", 35.0)["status"] == "applied"
    assert twin.get_zones("site_a")["zones"][0]["setpoint_c"] == 35.0


def test_assert_never_called_is_the_deny_path_assertion() -> None:
    """A denial that still forwards the call has denied nothing."""
    twin = FakeTwin()
    twin.assert_never_called("twin.apply_setpoint")

    twin.apply_setpoint("zone_north", 22.0)
    assert twin.tool_names_called == ["twin.apply_setpoint"]
    try:
        twin.assert_never_called("twin.apply_setpoint")
    except AssertionError as error:
        assert "reached the upstream twin" in str(error)
    else:
        raise AssertionError("assert_never_called did not fire on a forwarded call")


def test_default_zones_span_two_sites() -> None:
    """Cross-tenant tests need at least two tenants to be meaningful."""
    assert len({zone.site_id for zone in DEFAULT_ZONES}) >= 2

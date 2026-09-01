"""A twin that contains no physics, for testing the gateway against.

THIS FILE IS THE REPLACEABILITY PROOF. `docs/ARCHITECTURE.md` claims the
twin is swappable because the gateway only ever reaches it over MCP. The
whole gateway suite runs against THIS object rather than the real twin,
so a green suite with `cooling-twin` uninstalled is not an argument for
the claim, it is a demonstration of it.

It is deliberately dependency-free: no `mcp`, no `cooling_twin`, no
network. The gateway's proxy layer (M3) is what turns these method calls
into MCP tool calls, and the fake is injected at that seam. Keeping the
fake importable with nothing installed means the boundary suite runs on
a bare checkout.

The canned numbers are SHAPED like the real twin's but are not its
output and must never be quoted anywhere. `simulation_id` is a hash of
the request so identical inputs return an identical id, which is what
M2's caching requirement asks for and what a receipt binds to.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Matches policies/policy.yaml `safety.setpoint_c`. Duplicated here on
# purpose: the fake must be able to return an out-of-bounds value so
# tests can prove the GATEWAY rejects it rather than relying on the
# upstream to have been well behaved (ADR-0006, defence in depth).
SAFE_SETPOINT_MIN_C = 18.0
SAFE_SETPOINT_MAX_C = 27.0

MAX_SUMMARY_POINTS = 50


@dataclass(frozen=True)
class FakeZone:
    """One zone the fake twin claims to model."""

    zone_id: str
    site_id: str
    current_setpoint_c: float
    current_temperature_c: float


DEFAULT_ZONES: tuple[FakeZone, ...] = (
    FakeZone("zone_north", "site_a", 23.0, 23.4),
    FakeZone("zone_south", "site_a", 22.5, 22.9),
    FakeZone("zone_main", "site_b", 24.0, 24.2),
)


@dataclass
class FakeTwin:
    """Canned responses for every tool the real twin exposes.

    Attributes:
        zones: The zones this twin reports.
        calls: Every call made, in order, as `(tool_name, arguments)`.
            Tests assert on this to prove the gateway did NOT forward a
            denied request -- "the upstream was never called" is a
            security property and needs to be observable.
        fail_simulations: When true, simulations complete with an error
            status, so receipt validation check 7 can be exercised.
    """

    zones: tuple[FakeZone, ...] = DEFAULT_ZONES
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    fail_simulations: bool = False
    _applied: dict[str, float] = field(default_factory=dict)

    # -- READ tier ---------------------------------------------------- #

    def get_zones(self, site_id: str) -> dict[str, Any]:
        """Zones for one site, with their current state."""
        self.calls.append(("twin.get_zones", {"site_id": site_id}))
        return {
            "zones": [
                {
                    "zone_id": zone.zone_id,
                    "site_id": zone.site_id,
                    "setpoint_c": self._applied.get(zone.zone_id, zone.current_setpoint_c),
                    "zone_temperature_c": zone.current_temperature_c,
                }
                for zone in self.zones
                if zone.site_id == site_id
            ]
        }

    def get_sensor_history(self, zone_id: str, window_minutes: int) -> dict[str, Any]:
        """A SUMMARY, never a raw series.

        The real twin has 8,760 hourly points per year. Returning them
        would exhaust the model's context (T-12), so the contract is
        summary-plus-handle and the point cap is enforced here as well
        as at the gateway.
        """
        self.calls.append(
            ("twin.get_sensor_history", {"zone_id": zone_id, "window_minutes": window_minutes})
        )
        n_points = min(window_minutes // 60, MAX_SUMMARY_POINTS)
        return {
            "zone_id": zone_id,
            "window_minutes": window_minutes,
            "summary": {
                "mean_temperature_c": 23.1,
                "min_temperature_c": 21.8,
                "max_temperature_c": 24.6,
                "n_points": n_points,
            },
        }

    # -- SIMULATE tier ------------------------------------------------- #

    def start_simulation(
        self, zone_id: str, setpoint_c: float, horizon_hours: int
    ) -> dict[str, Any]:
        """Returns immediately with an id; never blocks (M2, ADR-0005).

        Deterministic id: identical inputs must produce an identical
        `simulation_id` so the cache can hit and so a receipt refers to
        one reproducible piece of work.
        """
        arguments = {"zone_id": zone_id, "setpoint_c": setpoint_c, "horizon_hours": horizon_hours}
        self.calls.append(("twin.start_simulation", arguments))
        digest = hashlib.sha256(json.dumps(arguments, sort_keys=True).encode()).hexdigest()
        return {"simulation_id": f"sim_{digest[:16]}", "status": "pending"}

    def get_simulation_result(self, simulation_id: str, detail: str = "summary") -> dict[str, Any]:
        """Completed result. `detail=summary` returns aggregates only.

        The real twin can still be running; the gateway must treat a
        `pending` status as a valid response, not an error, and must not
        mint a receipt for it (ADR-0004 mints on the first COMPLETED
        result, which is what makes validation check 7 meaningful).
        """
        self.calls.append(
            ("twin.get_simulation_result", {"simulation_id": simulation_id, "detail": detail})
        )
        if self.fail_simulations:
            return {
                "simulation_id": simulation_id,
                "status": "error",
                "error": "solver failed to converge",
            }
        return {
            "simulation_id": simulation_id,
            "status": "completed",
            "summary": {
                "predicted_energy_delta_pct": -3.1,
                "comfort_violation_hours": 0,
                "peak_load_kw": 1840.0,
            },
        }

    # -- APPLY tier ---------------------------------------------------- #

    def apply_setpoint(self, zone_id: str, setpoint_c: float) -> dict[str, Any]:
        """Writes to SIMULATED control state. No BMS is connected.

        The fake accepts values outside the safe range on purpose. If it
        refused them, a test could not tell whether the gateway enforced
        the bound or the upstream did -- and the gateway is the control
        point precisely because the upstream may be replaced by one that
        does not check (ADR-0006).
        """
        self.calls.append(("twin.apply_setpoint", {"zone_id": zone_id, "setpoint_c": setpoint_c}))
        self._applied[zone_id] = setpoint_c
        return {"zone_id": zone_id, "setpoint_c": setpoint_c, "status": "applied"}

    # -- test helpers -------------------------------------------------- #

    @property
    def tool_names_called(self) -> list[str]:
        """Just the tool names, for the common "was it forwarded?" assertion."""
        return [name for name, _arguments in self.calls]

    def assert_never_called(self, tool_name: str) -> None:
        """Fail if a tool reached the upstream at all.

        The single most important assertion in the adversarial suite: a
        denial that still forwards the call has denied nothing.
        """
        if tool_name in self.tool_names_called:
            raise AssertionError(
                f"{tool_name} reached the upstream twin. A gateway denial must "
                f"stop the call, not merely discard the response. Calls seen: "
                f"{self.tool_names_called}"
            )

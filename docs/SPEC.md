# Functional specification — MCP Policy Gateway

## Purpose

A security gateway between AI agents and a building cooling digital
twin. It authenticates the caller, classifies every tool into a
capability tier, and enforces the rule that no setpoint can be applied
unless the same principal has already simulated that exact change and
holds an unexpired, single-use receipt proving it.

## Principals and roles

A principal is derived **only** from a verified JWT. Nothing about
identity is ever read from a request body, a header the caller controls,
or a tool argument.

```
Principal(sub: str, roles: list[str], site_id: str)
```

| Role | read | simulate | apply |
|---|:--:|:--:|:--:|
| `viewer` | ✅ | ❌ | ❌ |
| `operator` | ✅ | ✅ | ❌ |
| `engineer` | ✅ | ✅ | ✅ |

`operator` is the interesting row: an authenticated caller with a
legitimate role, denied one specific tier. A gateway that only blocks
unknown callers has not demonstrated authorization.

## Tool contracts

Units are in every field name. `setpoint_c`, never `temperature`;
`horizon_hours`, never `duration`. This is T-13's mitigation, and it is
enforced by review rather than by a type.

### READ

```
twin.get_zones() -> {zones: [{zone_id, site_id, setpoint_c, zone_temperature_c}]}
```
`site_id` is injected from the principal and is not a caller argument.

```
twin.get_sensor_history(zone_id: str, window_minutes: int) -> {summary: {...}}
```
`window_minutes ≤ 10080`. Returns aggregates. **Never more than 50 raw
points** in any response.

### SIMULATE

```
twin.start_simulation(zone_id: str, setpoint_c: float, horizon_hours: int)
    -> {simulation_id: str, status: "pending"}
```
Returns immediately. `18.0 ≤ setpoint_c ≤ 27.0`, `horizon_hours ≤ 72`,
at most 3 concurrent per principal. Identical arguments **must** yield an
identical `simulation_id`.

```
twin.get_simulation_result(simulation_id: str, detail: "summary" | "slice")
    -> {status: "pending" | "completed" | "error", summary?: {...}}
```
`pending` is a valid response, not an error. On the **first** `completed`
response the gateway mints a receipt and returns it alongside the
summary.

### APPLY

```
twin.apply_setpoint(zone_id: str, setpoint_c: float, receipt: str)
    -> {zone_id, setpoint_c, status: "applied"}
```
Requires a receipt passing all seven checks
(`ARCHITECTURE.md#the-seven-checks`). Writes to **simulated control
state**; no BMS is connected.

## Decision outcomes

Every request terminates in exactly one outcome and emits exactly one
audit record. There is no third state.

| Outcome | Upstream called? |
|---|---|
| `allow` | yes |
| `deny` | **no** |

A denial that still forwards the call has denied nothing. This is
asserted directly by `FakeTwin.assert_never_called`.

## Reason codes

Denials carry a specific code. A log that cannot separate "expired" from
"replayed" cannot separate operator error from attack.

```
auth_missing_token        auth_invalid_signature    auth_expired
auth_wrong_audience       policy_unknown_tool       policy_no_tier
policy_role_denied        policy_constraint_failed  policy_limit_exceeded
receipt_missing           receipt_signature_invalid receipt_expired
receipt_replayed          receipt_subject_mismatch  receipt_site_mismatch
receipt_zone_mismatch     receipt_setpoint_mismatch ramp_limit_exceeded
upstream_error            internal_error
```

`internal_error` is a **deny**. An exception during evaluation must never
fail open.

## Non-functional requirements

| Requirement | Value |
|---|---|
| Deny by default | Unknown tool, unknown tier, failed evaluation, raised exception |
| Bypass flags | None. No debug mode, no admin override |
| Policy location | `policies/*.yaml`. No role, tool, zone, tier or bound in Python |
| Audit completeness | One record per request, denials included |
| Bounds enforcement | At the gateway as well as the twin (ADR-0006) |
| Secret handling | Never log a JWT, an `Authorization` header, or a receipt |
| Response size | Never more than 50 raw points to the model |
| Receipt TTL | 900 s |
| Setpoint range | 18.0–27.0 °C |
| Ramp limit | 2.0 °C/hour per zone |

## Out of scope

- Real BMS/BACnet actuation — the actuator is simulated
- OAuth device flow and token issuance — JWTs are pre-issued (ADR-0003)
- Multi-tenant key rotation
- Anything that would require the gateway to import the twin

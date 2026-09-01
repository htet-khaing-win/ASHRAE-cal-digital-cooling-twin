# MCP Policy Gateway · Building Cooling Control

> **Scope:** this file governs work under `services/`. The repository
> root `CLAUDE.md` still applies and remains the instructor control file
> for the twin. Where the two differ, this one wins **inside
> `services/`** and nowhere else.

## What this is

A security gateway between AI agents and a building cooling digital
twin. It authenticates callers, enforces capability-tier authorization
over tool calls, and requires proof of prior simulation before any
setpoint change is applied.

## Read before writing any code

| File | What it settles |
|---|---|
| `docs/SPEC.md` | Tool contracts, roles, reason codes |
| `docs/ARCHITECTURE.md` | Components, request lifecycle, receipt flow |
| `docs/THREAT-MODEL.md` | T-01…T-14, and what the receipt does *not* cover |
| `docs/MILESTONES.md` | The current milestone and its acceptance criteria |
| `docs/adr/` | Decisions already made — do not relitigate them |

## Non-negotiable rules

1. **DENY BY DEFAULT.** Unknown tool, unknown tier, failed evaluation,
   raised exception: all DENY. Never fail open.
2. **APPLY IS GATED.** `twin.apply_setpoint` must never reach the
   upstream without a receipt passing all seven checks. There is no
   bypass flag, no debug mode, no admin override. If you think you need
   one, stop and ask.
3. **Policy lives in `policies/*.yaml`.** Never hardcode a role, tool
   name, zone, tier or temperature bound in Python.
4. **Every request emits exactly one audit record**, including denials
   and receipt validation failures. A missing audit record is a bug of
   the same severity as a missing authorization check.
5. **Physical bounds are enforced at the gateway**, not only at the twin.
   The twin may be replaced; the gateway is the control point (ADR-0007).
6. **Units in every schema field name:** `setpoint_c`, `horizon_hours`,
   `window_minutes`. Never a bare `temperature` or `duration`.
7. **Never log a raw JWT, an `Authorization` header, or a receipt.**
8. **Summarise simulation results.** Never return more than 50 raw points.
9. **`gateway/` must never import `cooling_twin` or `twin_mcp`.**
   `tests/test_boundary.py` enforces this. Changing it is an ADR-0007
   decision, not a test edit.

## A denial must stop the call

A denial that forwards the request and discards the response has denied
nothing. Every deny-path test asserts
`fake_twin.assert_never_called("twin.apply_setpoint")`.

## Verify before you claim done

Run from the repository root, in the `cooling-twin` conda environment:

```bash
make gateway-lint     # ruff + mypy over services/
make gateway-test     # pytest services/
make all              # the whole repo: twin + dashboard + gateway
```

There is no `uv` in this repository. `SCOPE-UPDATE.md` proposed it; the
conda environment is the established path and two package managers in
one repo buy nothing. See `environment.yml`.

## Style

- Python 3.11+, async at the transport boundary
- Pydantic models for every boundary object: `Principal`, `ToolRef`,
  `Decision`, `Receipt`, `SimulationSummary`
- No bare `except:`. Catch specific exceptions
- Functions under 40 lines
- Full type hints and Google-style docstrings, as in `src/cooling_twin`

## Anti-patterns that will be rejected

- Receipt validation logic anywhere other than `gateway/receipts.py`
- Any code path where apply succeeds without receipt verification
- Policy logic written as `if`/`elif` chains
- Tests that only assert the happy path
- Returning full simulation timeseries into the model context
- New dependencies added without an ADR
- Claiming the system actuates a physical building — it does not; the
  actuator is simulated and `docs/ARCHITECTURE.md` says so

## Current state

> **Milestone: M0 complete.** Scaffold, boundary test, fake twin, policy
> v2, docs and seven ADRs are in place. No gateway logic written yet.
> Next: M1 (twin read tools). See `docs/MILESTONES.md`.

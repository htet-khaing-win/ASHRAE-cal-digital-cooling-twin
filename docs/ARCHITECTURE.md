# Architecture

## The one sentence

An agent cannot change a building's setpoint without first modelling the
consequence, and the gateway — not the agent's good intentions — is what
makes that true.

## Components

```
┌─────────┐   MCP over        ┌──────────────┐   MCP over      ┌──────────┐
│  AGENT  │ ─ streamable ───▶ │   GATEWAY    │ ─ HTTP ───────▶ │ twin_mcp │
│         │   HTTP            │              │                 │          │
│ untrust-│                   │ auth         │                 │ adapter  │
│ ed      │ ◀───────────────  │ policy       │ ◀────────────── │          │
└─────────┘  result + receipt │ receipts     │                 └────┬─────┘
                              │ audit        │                      │ python
                              └──────┬───────┘                      │ import
                                     │                              ▼
                                     ▼                       ┌──────────────┐
                              ┌──────────┐                   │ cooling_twin │
                              │  REDIS   │                   │  RC network  │
                              │ sim cache│                   │  + DOE-2     │
                              │ nonces   │                   │  curves      │
                              │ ramp state                   └──────────────┘
                              └──────────┘
```

**The gateway never imports the twin.** It reaches it only over MCP.
That is what makes "the twin may be replaced; the gateway is the control
point" a fact rather than a slogan, and
`services/gateway/tests/test_boundary.py` fails the build if it stops
being true. `services/gateway/tests/fakes/fake_twin.py` is the
demonstration: the entire gateway suite runs against a twin with no
physics in it.

| Package | May import `cooling_twin`? | Role |
|---|---|---|
| `src/cooling_twin` | — | The physics. Knows nothing of identity, MCP, or authorization |
| `services/twin_mcp` | **yes** | Adapter. Physics on one side, an MCP tool contract on the other |
| `services/gateway` | **never** | Control point. Auth, tiers, receipts, audit |

## Request lifecycle

Every request takes exactly this path. There is no branch that skips a
step, and every terminal state emits exactly one audit record.

```
 1. transport      receive MCP tools/call
 2. authenticate   verify JWT → Principal(sub, roles, site_id)     ─┐
 3. resolve        look up the tool's rule in policy.yaml           │ any failure
 4. tier           rule declares a tier, or DENY                    │ here is a
 5. authorize      principal's role ∈ rule.allow_roles, or DENY     │ DENY with a
 6. constrain      validate arguments against rule.constraints      │ distinct
 7. inject         overwrite site_id from the principal             │ reason code,
 8. limits         per-principal concurrency / ramp checks          │ and NO
 9. receipt        apply tier only: seven checks (see below)        │ upstream
10. forward        call twin_mcp                                   ─┘ call
11. summarise      never return >50 raw points
12. mint           on a completed simulation result, mint a receipt
13. audit          exactly one record, denials included
```

Step 10 is the only step that touches the upstream. **A denial must stop
the call, not discard the response** — `FakeTwin.assert_never_called` is
how the suite proves it.

## The receipt flow

```
  agent                     gateway                      twin
    │                          │                           │
    │ start_simulation ───────▶│ policy ✓ ─────────────────▶│
    │◀──────────── simulation_id (pending)                  │
    │                          │                           │
    │ get_simulation_result ──▶│ policy ✓ ─────────────────▶│
    │                          │◀────── status: completed   │
    │                          │ MINT receipt ◀── HMAC key  │
    │◀───── summary + receipt  │  binds subject, site_id,   │
    │                          │  zone_id, setpoint_c,      │
    │                          │  nonce, exp                │
    │                          │                           │
    │ apply_setpoint ─────────▶│ policy ✓                   │
    │        + receipt         │ 7 checks ✓                 │
    │                          │ consume nonce ────────────▶│
    │◀──────────── applied     │                           │
    │                          │                           │
    │ apply_setpoint ─────────▶│ ✗ DENY receipt_replayed    │
    │        + same receipt    │   (upstream NOT called)    │
```

**Minting happens on the first completed `get_simulation_result`, not on
`start_simulation`.** Simulations are asynchronous: `start_simulation`
returns immediately with a pending job, so at that moment validation
check 7 ("the simulation completed without error") cannot be evaluated.
Minting at result-fetch time makes check 7 a mint-time invariant and
binds the receipt to a result the agent has actually seen. See
`adr/0004-simulation-receipts.md` and `adr/0005-async-simulation-jobs.md`.

### The seven checks

All must pass. Each failure has its own reason code, because an audit
log that cannot distinguish "expired" from "replayed" from "wrong zone"
cannot tell an operator error from an attack.

| # | Check | Reason code on failure | Threat |
|---|---|---|---|
| 1 | HMAC signature verifies | `receipt_signature_invalid` | T-10 |
| 2 | Not expired (TTL 900 s) | `receipt_expired` | T-09 |
| 3 | Nonce not consumed | `receipt_replayed` | T-09 |
| 4 | `subject` matches principal | `receipt_subject_mismatch` | T-10 |
| 5 | `site_id` matches argument | `receipt_site_mismatch` | T-10 |
| 6 | `zone_id` matches argument | `receipt_zone_mismatch` | T-10 |
| 7 | `setpoint_c` within tolerance | `receipt_setpoint_mismatch` | T-08 |

Check 5 exists because zone ids may collide across sites; binding only
the subject leaves a principal with two-site access able to move a
receipt between them.

Receipt verification lives in **`gateway/receipts.py` and nowhere else**.
Scattering it across `proxy.py` would make "every path to apply is
validated" unprovable by reading, which is the only way anyone can be
confident of it.

## State that outlives a request

Three things, all in Redis, all with the same TTL discipline:

| State | Why it cannot be in-process |
|---|---|
| Simulation cache | M2 requires identical inputs to return an identical id and a cached result |
| Consumed nonces | Single-use is meaningless across two replicas if each holds its own set |
| Per-zone last applied setpoint + timestamp | The ramp limit is a rate over time, so it needs history |

M8 runs **two gateway replicas**. In-memory state and two replicas are
incompatible — a nonce consumed on pod A replays on pod B, defeating
T-09 — so Redis lands in **M2**, not M8. See
`adr/0005-async-simulation-jobs.md`.

## Scope: the actuator is simulated

`twin.apply_setpoint` writes to the twin's **simulated control state**.
No building management system is connected. This project is a
historical-replay twin over 2016–2017 BDG2 data, and the repository
README's "advisory mode, offline validation only" governs here too.

This costs the security design nothing. Every control — tiers, bounds,
receipts, single-use nonces, audit — is exercised exactly as it would be
against a real actuator, and the twin is replaceable by construction. It
does mean one thing must never be written: that this system controls a
physical building. It does not.

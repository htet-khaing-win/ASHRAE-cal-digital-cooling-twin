# ADR-0004 — Simulation receipts, minted by the gateway

**Status:** accepted · **Date:** 2026-09-01 · **Milestone:** M6

> This is the project's central decision. Everything else exists to
> support or verify it.

## Context

An agent with `engineer` role can call `twin.apply_setpoint`. Role
checks and argument bounds establish that it is *allowed* to and that
the value is *in range*. Neither establishes that anyone modelled what
the change would do.

"The agent should simulate first" is a wish. The requirement is that it
**cannot** apply without having simulated.

## Decision

When the gateway returns a **completed** simulation result, it mints a
receipt: an HMAC-signed token binding

```
subject · site_id · zone_id · setpoint_c · simulation_id · nonce · exp
```

and returns it alongside the summary. `apply_setpoint` requires that
receipt, and the gateway verifies it and consumes the nonce.

### Why the gateway mints it, not the twin

This is the question an interviewer will ask, so it is answered here
first.

The twin is a physics model. It has no notion of identity, of
authorization, or of replay. Giving it receipt-minting would mean:

- the twin must learn what a principal is, so authentication leaks
  downstream into a component that should be swappable
- the nonce store becomes the twin's problem, so the twin becomes
  stateful and un-replaceable
- the security property becomes untestable against a fake twin —
  `fakes/fake_twin.py` would have to reimplement the security model to
  be useful, at which point it is testing itself
- and the gateway becomes **decorative**: a proxy that forwards calls
  while the real control lives elsewhere

Keeping it in the gateway is what makes the gateway load-bearing. The
twin stays a thing that answers "what would happen if"; the gateway
stays the thing that decides who may ask and who may act.

### Why minting happens at result-fetch, not at start

`SCOPE-UPDATE.md` originally specified minting when `start_simulation`
"returns successfully". That does not work, and the reason is worth
recording.

Simulations are asynchronous (ADR-0005): `start_simulation` returns
immediately with `status: pending`. At that moment validation check 7 —
*the simulation it refers to completed without error* — cannot be
evaluated, because it has not completed. Minting there would force the
gateway to call back to the twin at apply time to find out, adding a
network hop inside the deny path and a failure mode where apply is
denied because the twin is unreachable.

Minting on the first **completed** `get_simulation_result` instead makes
check 7 a mint-time invariant: a receipt cannot exist for a failed or
unfinished simulation. It also binds the receipt to a result the agent
has actually *seen*, which is a stronger statement about modelling than
"a job was started".

### The seven checks

| # | Check | Reason code | Prevents |
|---|---|---|---|
| 1 | HMAC signature verifies | `receipt_signature_invalid` | forged receipts |
| 2 | Not expired (TTL 900 s) | `receipt_expired` | indefinitely valid authority |
| 3 | Nonce not consumed | `receipt_replayed` | one simulation, many applies (T-09) |
| 4 | `subject` matches principal | `receipt_subject_mismatch` | using another's receipt (T-10) |
| 5 | `site_id` matches argument | `receipt_site_mismatch` | cross-tenant reuse (T-10) |
| 6 | `zone_id` matches argument | `receipt_zone_mismatch` | simulate A, apply B (T-10) |
| 7 | `setpoint_c` within tolerance | `receipt_setpoint_mismatch` | simulate 22, apply 26 (T-08) |

**Check 5 was not in the original draft.** It binds only after noticing
that the policy injects `site_id` on both the simulate and apply rules
while the receipt bound only `[subject, zone_id, setpoint_c]`. Zone ids
may collide across sites, so a principal with access to two sites could
move a receipt between them — precisely T-10.

Each failure gets its **own** code because an audit log that cannot
distinguish "expired" from "replayed" from "wrong zone" cannot
distinguish an operator's mistake from an attack in progress.

## What this does NOT do

Stated here, not discovered later.

> **The receipt proves modelling happened. It does not prove the model
> approved.**

An agent may simulate a setpoint, receive a summary showing comfort
violations, and apply it anyway — all seven checks pass. It is a
**procedural** control, not an **outcome** control. Prompt injection is
the same gap: an injected instruction that persuades the agent to
simulate-then-apply produces a perfectly valid receipt.

Closing it is M6b (outcome predicates in policy). It is deliberately a
separate milestone: the procedural control is complete and testable on
its own, and conflating the two would make neither demonstrable.

The receipt also does nothing for T-11 (bounds), T-12 (context flooding),
T-13 (units) or T-14 (cost) — those are policy constraints, and claiming
otherwise would overstate the mechanism.

## On the receipt travelling through model context

By design the receipt is returned to the agent and therefore enters the
model's context window; it may reach traces, transcripts and eval logs.
This is safe, and the reason is the design rather than an accident:

**A receipt confers no authority its holder does not already have.** Only
`engineer` may call `apply_setpoint` at all. The receipt is
subject-bound, single-use, and expires in 15 minutes. Whoever holds a
leaked receipt is either already an engineer for that site — and could
mint their own by simulating — or cannot use it.

It is a capability token deliberately handed to an untrusted planner,
and the bindings are what make that defensible. Rule 7 ("never log a
receipt") still applies: context exposure is accepted, *logging* it is
gratuitous additional exposure with no compensating benefit.

## Alternatives

**Twin mints the receipt.** See above. It makes the gateway decorative.

**Server-side session state: "this principal simulated X recently".**
No token at all; the gateway remembers. Genuinely simpler and avoids
signing. Rejected because it cannot bind the *specific* simulation the
agent looked at — the agent may have run three, and "recently simulated
something for this zone" is a much weaker claim than "holds proof of
this exact result". It also makes the audit trail worse: there is
nothing to log that identifies which modelling justified which action.

**Require the agent to pass `simulation_id` and have the gateway
re-fetch it.** No crypto, and the binding is to a real result. Rejected:
it puts a network call to the twin inside the deny path, so the twin
being slow or down turns into apply being denied — and worse, into a
temptation to cache or fail open. It also gives no replay protection
without adding a consumed-id store anyway, at which point the HMAC is
nearly free.

**Longer TTL, or no expiry.** Rejected: a receipt is a statement that
the world was modelled *recently*. Weather moves; a 12-hour-old
simulation is a claim about a building that no longer exists.

## Consequences

- Receipt verification lives in `gateway/receipts.py` and **nowhere
  else**. Scattering it across `proxy.py` makes "every path to apply is
  validated" unprovable by reading, which is the only way anyone can be
  confident of it.
- Consumed nonces need a store with at least the receipt's TTL, shared
  across replicas (ADR-0005).
- The HMAC key is a Kubernetes Secret, never in config or an image.
- `setpoint_match_tolerance_c` is an attack surface: repeated
  simulate-then-apply within tolerance walks the setpoint. The **ramp
  limit** is what bounds that, which is why it is stateful too.

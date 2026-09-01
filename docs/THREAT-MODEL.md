# Threat model — MCP Policy Gateway

Every threat here has an ID, and every ID has a test in
`services/gateway/tests/test_redteam.py` named after it (M9). A threat
without a test is an intention; a test asserting an attack *fails* is a
control.

**Scope.** The attacker is a capable agent — or a human driving one —
that holds valid credentials for *some* role, plus anyone who can inject
text into that agent's context. Compromise of the gateway host, the HMAC
signing key, or the Kubernetes control plane is out of scope: those
defeat the design rather than testing it.

**Trust boundaries.**

```
 untrusted            semi-trusted             trusted
 ─────────            ────────────             ───────
 tool results   →     the AGENT          →     the GATEWAY   →   twin_mcp
 web content          (may be steered           (control          (physics,
 user prompts          by injection)             point)            no identity)
```

The agent is **not** trusted. That is the premise of the whole project:
the gateway enforces what the agent may do rather than trusting it to
behave, and the receipt mechanism exists because "the agent should
simulate first" is a wish while "the agent cannot apply without a
receipt" is a control.

---

## Threats

| ID | Threat | Attack scenario | Mitigation | Milestone |
|---|---|---|---|---|
| T-01 | Unauthenticated access | Caller presents no token, or a malformed one | Reject at the transport boundary; no upstream call on any failure path | M4 |
| T-02 | Forged token | Attacker signs a token with their own key, or uses `alg: none` | Verify signature against the configured key; reject unexpected algorithms explicitly | M4 |
| T-03 | Expired / replayed token | A captured token is reused after its lifetime | Verify `exp`, `nbf`, `aud`, `iss` | M4 |
| T-04 | Role escalation via claims | Caller edits `roles` in an unsigned or weakly-checked token | Roles read only from the verified token; never from request body or headers | M4 |
| T-05 | Unlisted tool invocation | Caller invokes a tool absent from policy, or one added upstream but not declared | Deny by default; an unlisted tool and an untiered tool are both denied | M5 |
| T-06 | Cross-tenant data access | Caller requests another site's zones or history | `site_id` injected from the principal, never accepted from the caller | M5 |
| T-07 | Argument tampering | Out-of-range or wrong-typed arguments smuggled past validation | Constraints declared per rule in policy, evaluated before the upstream call | M5 |
| T-08 | **Unsimulated actuation** | Agent calls `apply_setpoint` having never modelled the consequence | Receipt required; no receipt is a deny | M6 |
| T-09 | **Receipt replay** | Agent reuses one receipt to apply repeatedly, or an attacker captures and replays it | Single-use nonce, consumed on first successful apply | M6 |
| T-10 | **Receipt substitution** | A receipt minted for zone A (or site A, or principal A) is presented elsewhere | Receipt binds `subject`, `site_id`, `zone_id`, `setpoint_c`; all four checked | M6 |
| T-11 | Actuation bounds bypass | Setpoint outside the safe range, or a ramp rate that stresses equipment | Bounds and ramp limits in policy, enforced at the gateway as well as the twin | M5 / M6 |
| T-12 | Context flooding | A long-horizon simulation returns thousands of timesteps, exhausting context or inflating cost | Summary-plus-handle response shape; detail fetched in bounded slices, ≤50 points | M2 |
| T-13 | Unit confusion | Setpoint interpreted as Fahrenheit when the twin expects Celsius | Units encoded in field names; range validation rejects out-of-band values | M1 |
| T-14 | Simulation cost abuse | Agent loop starts hundreds of simulations, exhausting compute | Per-principal concurrent simulation cap in policy | M5 |

Bold rows are the receipt mechanism's own threats. See
`adr/0004-simulation-receipts.md`.

---

## What the receipt does NOT mitigate

Stated here rather than discovered later, because the gap is the most
interesting question anyone will ask about this design.

> **A receipt proves that modelling happened. It does not prove the model
> approved.**

An agent may simulate a harmful setpoint, receive a result showing
comfort violations or an energy increase, and apply it anyway. Every one
of the seven validation checks passes: the principal did simulate that
exact change, recently, once. The receipt is a **procedural** control,
not an **outcome** control.

The same gap covers prompt injection. An injected instruction that
persuades the agent to simulate-then-apply a harmful change produces a
perfectly valid receipt. The gateway constrains the *shape* of the
agent's actions, not its judgment.

Closing it requires an **outcome predicate** — a policy assertion over
the simulation summary the receipt refers to:

```yaml
require_receipt:
  outcome_constraints:
    comfort_violation_hours: { max: 0 }
    predicted_energy_delta_pct: { max: 5.0 }
```

This is tracked as **M6b** and is worth doing: it turns an authorization
gateway into a safety gateway, and the twin already produces uncertainty
intervals, so the predicate can be bound to the *upper* bound of a
predicted change rather than its point estimate.

Two further limits worth naming:

- **T-11 is not the receipt's work.** Bounds are enforced by policy
  constraints; a receipt for an in-range setpoint says nothing about
  whether that range was correct.
- **A leaked receipt is not an escalation.** It travels through the
  agent's context by design and may reach traces and logs. It is safe
  because it confers no authority its holder lacks: only `engineer` may
  apply at all, and the receipt is subject-bound, single-use and
  TTL-limited. An attacker holding one is either already an engineer for
  that site — and could mint their own — or cannot use it.

---

## Residual risks accepted

| Risk | Why accepted |
|---|---|
| Gateway host or HMAC key compromise | Defeats the design rather than testing it; mitigated operationally (Secret mount, non-root, read-only rootfs) |
| Malicious `engineer` | The role is defined as trusted to actuate. The audit log is the control, not prevention |
| Twin returning wrong physics | Out of scope for the gateway. The twin's own validation is ASHRAE G14 on a held-out year (see the repository README) |
| Clock skew across replicas | TTL comparisons use the gateway's clock; skew beyond the TTL would need NTP failure plus a captured receipt |

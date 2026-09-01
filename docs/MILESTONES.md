# Milestones

Each milestone is independently demoable. Do not start N+1 until N passes.

Effort figures are revised from `SCOPE-UPDATE.md` §6: that document
assumed the twin was service-ready ("wrapping an existing model"). It is
not — it is a calibrated physics model plus an offline batch pipeline
with no zone abstraction, no job runner, no control state and no
actuation concept. M1, M2 and M10 are budgeted accordingly.

---

## M0 — Scaffold ✅

- [x] `services/` layout with three distributions, twin unchanged
- [x] Gateway declares no dependency on the twin
- [x] `test_boundary.py` — AST scan, declared deps, runtime probe
- [x] `fakes/fake_twin.py` — the replaceability demonstration
- [x] `policies/policy.yaml` v2
- [x] `docs/` — SPEC, ARCHITECTURE, THREAT-MODEL, MILESTONES, 7 ADRs
- [x] CI split into `twin` / `gateway` / `twin-conda` jobs
- [x] `services/CLAUDE.md` (root `CLAUDE.md` kept as the teaching file)

## M1 — Twin MCP server, read tools — *2.5 d*

- [ ] `twin.get_zones` returns zone list with current state
- [ ] `twin.get_sensor_history(zone_id, window_minutes)` returns a
      summary, never more than 50 raw points
- [ ] All schema fields carry units in their names
- [ ] Inspector can list and call both tools

> **The twin has no zone concept.** `rc.py` is a single lumped 2R2C node
> and `config/buildings.yaml` holds *buildings*. Decide and document
> whether a zone is a building or a sub-node; do not let the word "zone"
> imply a spatial resolution the physics does not have.

## M2 — Simulation as an async job — *4 d*

- [ ] `twin.start_simulation` returns a `simulation_id` immediately
- [ ] `twin.get_simulation_result(simulation_id, detail)` polls
- [ ] `detail=summary` returns aggregates; `detail=slice` a bounded window
- [ ] Identical inputs produce an identical id and a cached result
- [ ] A running simulation returns `pending`, not a block
- [ ] **Redis** backs the cache, consumed nonces and ramp state (ADR-0005)

> `counterfactual.py` evaluates a scenario over a whole year in one call.
> A bounded `horizon_hours ≤ 72` run is a different call pattern.

## M3 — Gateway pass-through — *2 d*

- [ ] Tools exposed with a `twin.` prefix
- [ ] `tools/list` and `tools/call` round-trip correctly
- [ ] No auth yet. Transport correctness only

## M4 — Authentication — *1.5 d*

- [ ] Missing, malformed, expired, wrong-audience, wrong-key → 401
- [ ] Valid token yields `Principal(sub, roles, site_id)`
- [ ] **No upstream call on any failure path**

## M5 — Capability tiers and policy — *3 d*

- [ ] `policy.yaml` v2 drives every decision
- [ ] Unlisted tool denied; untiered tool denied
- [ ] `operator` can simulate, cannot apply
- [ ] Setpoint outside 18–27 °C denied at the gateway, before upstream
- [ ] Concurrent simulation cap enforced per principal

## M6 — Simulation receipts — *3 d*

- [ ] Receipt minted on the first **completed** `get_simulation_result`
- [ ] `apply_setpoint` without a receipt is denied
- [ ] All seven checks implemented, each with a distinct reason code
- [ ] Replay of a consumed receipt denied
- [ ] Receipt from zone A rejected on zone B
- [ ] Receipt from site A rejected on site B
- [ ] Receipt from principal A rejected for principal B
- [ ] Ramp limit enforced against per-zone applied history

> Write `gateway/receipts.py` with its tests **before** touching
> `proxy.py`. Verification lives in that module and nowhere else.

## M6b — Outcome predicates — *1.5 d, recommended*

- [ ] Policy may assert over the simulation summary a receipt refers to
- [ ] Apply denied when the modelled outcome violates the predicate
- [ ] Predicate may bind the **upper** uncertainty bound, not the point estimate

> Closes the gap in THREAT-MODEL.md's "What the receipt does NOT
> mitigate": today a receipt proves modelling happened, not that the
> model approved. This is what turns an authorization gateway into a
> safety gateway.

## M7 — Audit logging — *1 d*

- [ ] One record per request, denials included
- [ ] Receipt failures logged with their specific reason code
- [ ] `correlation_id` links gateway record to upstream call
- [ ] No secrets, tokens or receipts in any record

## M8 — Packaging and Kubernetes — *2.5 d*

- [ ] `docker compose up` brings the whole system up
- [ ] kind cluster runs gateway (2 replicas) and twin
- [ ] Policy mounted as ConfigMap, hot-reloadable
- [ ] Receipt HMAC key mounted as Secret
- [ ] Non-root, `readOnlyRootFilesystem`, dropped capabilities

## M9 — Adversarial test suite — *2.5 d*

- [ ] One test per threat ID T-01 … T-14
- [ ] Test names carry their threat ID
- [ ] Every test asserts an attack **fails**
- [ ] Every denial test asserts the upstream was never called

## M10 — Closed-loop evaluation — *6.5 d*

- [ ] Rule-based baseline controller
- [ ] Agent-driven episodes against the same scenarios
- [ ] Metrics: energy proxy, comfort violations, tool calls per episode,
      **simulations per apply**
- [ ] Results table with episode count stated honestly

> **Pre-register the claim before running this.** The twin's
> `t_setpoint_c` is *unidentified*: 22.90–26.00 °C all fit within 5% of
> the calibrated objective (Q9), the counterfactual intervenes on a
> fitted parameter rather than a measured thermostat, and one scenario's
> equifinality band already reaches 4× its point estimate and contains
> zero. So:
>
> - ✅ "Agent and baseline are scored on the **same twin**, so its error
>   is common-mode and the comparison is valid in relative terms."
> - ✅ "The agent reaches the same outcome with fewer simulations."
> - ❌ "The agent saves X% energy."
>
> **`simulations per apply` is the headline metric.** It measures a
> gateway property and does not depend on the physics being right.

---

| Milestone | Days | Cut point |
|---|---|---|
| M0–M7 | 18.5 | **Minimum shippable** — demos well, write it up |
| + M8, M9 | 23.5 | **Interview-ready** — the version for the CV |
| + M10 | 30 | **Differentiated** — treat as a follow-up post, not a launch blocker |
| + docs, ADRs, demo video | 33 | |

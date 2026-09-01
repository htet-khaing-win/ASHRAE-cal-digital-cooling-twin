# ADR-0005 — Async simulation jobs, and Redis in M2 rather than M8

**Status:** accepted · **Date:** 2026-09-01 · **Milestone:** M2

## Context

A 72-hour simulation on the calibrated twin takes seconds; a longer or
concurrent one takes longer. Two questions follow: does the tool block,
and where does the state live?

## Decision

1. **Start/poll**, not a blocking call. `start_simulation` returns a
   `simulation_id` immediately; `get_simulation_result` polls and may
   return `pending`.
2. **Redis from M2**, not M8. It backs three things: the simulation
   cache, consumed receipt nonces, and per-zone applied-setpoint history.

## Why start/poll

**A blocking call** is simpler and needs no job store. Rejected: it puts
an unbounded wait inside an MCP tool call, which means client timeouts
become the de-facto policy, and a timeout mid-simulation leaves the agent
unable to tell "still running" from "failed". It also makes the
concurrency cap (T-14) unenforceable in any useful way — you cannot cap
what you cannot see queued.

**Streaming progress over SSE.** Attractive, and MCP supports it.
Deferred: it does not remove the need for a job store (a reconnecting
client must still find its result) and it adds a second delivery path to
the same data. Poll first; stream later if the wait is genuinely a UX
problem.

Making `pending` an ordinary, valid response rather than an error matters
more than it looks: an agent that treats "not finished" as failure will
retry, which is how a cost-abuse loop starts.

## Why Redis in M2, not M8

`SCOPE-UPDATE.md` §6 named this as the most likely way the schedule
slips, and then left the decision open. Closing it now:

M8 requires **two gateway replicas**. In-memory state and two replicas
are incompatible in a way that is not a performance issue but a security
failure:

```
  agent → pod A: apply(receipt R)  → nonce R consumed in pod A's memory ✓
  agent → pod B: apply(receipt R)  → pod B has never seen R          ✗ APPLIED
```

That defeats T-09 (replay) completely, and it would be discovered in
week 4 while wiring Kubernetes, by which point receipts, audit and the
red-team suite have all been built against the wrong assumption.

The alternative — accept single-replica for the MVP and record it — was
considered. Rejected because M8's own acceptance criteria already say two
replicas, so the debt comes due inside the plan rather than after it.

M2 needs a cache regardless ("identical inputs produce an identical
`simulation_id` and a cached result"), so one dependency arrives once and
serves three needs:

| State | TTL | Why shared |
|---|---|---|
| Simulation cache | hours | Identical inputs → identical id and result |
| Consumed nonces | ≥ receipt TTL (900 s) | Single-use is meaningless per-replica |
| Per-zone last applied setpoint + timestamp | rolling window | The ramp limit is a rate over time |

**The third row is the one nobody counted.** `max_ramp_c_per_hour` cannot
be evaluated from a single request; it needs to know what was applied to
that zone and when. It is cross-request state exactly like the nonce
store, and it was absent from the scope document's list.

## Alternatives for the store

**In-process dict + single replica.** Simplest. Rejected above.

**Postgres.** Already durable, already transactional, and a natural fit
for the audit log. Rejected for this state: all three items are
TTL-bounded and none needs durability across a full outage — a lost
nonce set after a total cluster restart fails *closed* (receipts expire
in 15 minutes anyway). Redis's native TTL expresses that directly; in
Postgres it is a sweeper job.

**Sticky sessions at the ingress.** Would let in-memory state survive
two replicas. Rejected: it makes a security invariant depend on load
balancer configuration, and it fails during a rolling deploy — which is
exactly when a replay would go unnoticed.

## Consequences

- `docker compose` and the kind manifests carry Redis from M2 onward
- Nonce consumption must be **atomic** — a compare-and-set, not
  read-then-write, or two concurrent applies race one nonce (an explicit
  M9 red-team case)
- A Redis outage must fail **closed**: no store, no apply
- Local development needs Redis running, so the test suite must use a
  fake or a fixture rather than requiring it

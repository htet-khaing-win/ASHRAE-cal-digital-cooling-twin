# ADR-0006 — Summary-plus-handle responses, never raw timeseries

**Status:** accepted · **Date:** 2026-09-01 · **Milestone:** M2

## Context
The twin produces 8,760 hourly points per simulated year. An agent that
receives them pays for them in context and in money, and a long-horizon
request becomes a denial-of-service against the model itself (T-12).

## Decision
Every simulation response is a **summary** plus a handle. Detail is
fetched in bounded slices, capped at 50 points per response, enforced at
the gateway as well as the twin.

## Alternatives
**Return the full series and let the agent sample.** Simple, and the
agent could do its own analysis. Rejected: the cost is paid on the way
in, before the agent can decide it did not want the data. A 72-hour
request is fine; the cap exists for the request that is not.

**Server-side pagination with a cursor.** Nearly the same thing and
slightly more general. Deferred: `detail=slice` with an explicit window
is easier to bound in policy, and a cursor is state to expire.

**Compress or downsample adaptively.** Rejected for the MVP: an
adaptively downsampled series is a number the agent cannot trace, and
this project's standard is that every number traces to an artifact.

## Consequences
- The summary's fields become an API contract; adding one is a change
- 50 is a policy constant (`max_points`), not a literal in Python
- Outcome predicates (M6b) read the summary, so its shape matters twice

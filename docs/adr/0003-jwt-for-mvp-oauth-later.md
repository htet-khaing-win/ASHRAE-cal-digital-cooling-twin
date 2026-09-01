# ADR-0003 — Pre-issued JWTs for the MVP; OAuth deferred

**Status:** accepted · **Date:** 2026-09-01 · **Milestone:** M4

## Context
The gateway needs an authenticated principal with a subject, roles and a
site. It does not need to *issue* credentials.

## Decision
Verify pre-issued JWTs (signature, `exp`, `nbf`, `aud`, `iss`, explicit
algorithm allow-list). `Principal` is built only from verified claims.

## Alternatives
**Full OAuth 2.1 with the MCP authorization spec.** The correct
production answer and where this should go. Rejected for the MVP: an
authorization server, token endpoint, consent and refresh flows are
several days that demonstrate no property this project is about. The
migration path is narrow by design — everything downstream depends on
`Principal`, not on how it was obtained.

**mTLS.** Strong, and appropriate inside a cluster. Rejected: it
authenticates a *workload*, not a user, and the role split this project
demonstrates is per-principal.

**API keys.** Rejected: no standard claim structure, so roles and site
would need a side lookup, and revocation becomes a bespoke problem.

## Consequences
- Token issuance is out of scope and must be said so in the README
- `alg: none` and algorithm-confusion attacks are explicit test cases (T-02)
- Swapping in OAuth later touches `auth.py` only

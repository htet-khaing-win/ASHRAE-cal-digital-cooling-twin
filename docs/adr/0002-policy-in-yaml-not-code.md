# ADR-0002 — Policy as YAML data, not code and not OPA

**Status:** accepted · **Date:** 2026-09-01 · **Milestone:** M5

## Context
Authorization must be reviewable by someone who does not read Python,
and hot-reloadable from a ConfigMap without a redeploy (M8).

## Decision
`policies/policy.yaml`, version 2, evaluated by a small interpreter in
`gateway/policy.py`. No role, tool name, zone, tier or bound appears in
Python.

## Alternatives
**OPA / Rego.** The industry answer, genuinely better at scale: a real
policy language, decision logs, existing tooling. Rejected for the MVP
because it adds a sidecar, a second language, and a network hop inside
the deny path — and the ruleset here is ~5 rules over 3 roles. The cost
is real and the benefit does not arrive until the ruleset outgrows one
screen. Revisit when rules exceed ~30 or when policy must be authored by
someone outside this repo.

**Cedar.** Better ergonomics than Rego and an embeddable evaluator.
Rejected mainly on ecosystem maturity in Python at time of writing.

**if/elif chains in Python.** Rejected outright. It makes the answer to
"what may an operator do?" a code-reading exercise, and it puts the
authorization surface behind a rebuild.

## Consequences
- A small evaluator to write and test, including its failure modes
- Every unknown construct in the YAML must be a **deny**, not a skip
- Schema-validate the policy at load; a malformed policy fails startup
  rather than silently authorizing nothing or everything

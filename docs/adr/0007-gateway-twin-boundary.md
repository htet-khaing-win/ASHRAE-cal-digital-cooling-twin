# ADR-0007 — Bounds at the gateway and the twin; the gateway never imports the twin

**Status:** accepted · **Date:** 2026-09-01 · **Milestone:** M0, M5

## Context
Two questions with one answer. Where are physical bounds enforced? And
what stops the gateway from becoming coupled to this particular twin?

## Decision
1. Bounds are enforced at the **gateway** and again at the twin.
2. `services/gateway` **never** imports `cooling_twin` or `twin_mcp`,
   and `tests/test_boundary.py` fails the build if it does.

## Why duplicate the bound
The obvious objection is that duplicated validation drifts. It does —
and that is cheaper than the alternative. The twin is replaceable by
design; the replacement may not validate anything. Enforcing only
upstream would mean the safety property lives in the component the
architecture says is swappable. The gateway is the control point, so the
bound lives at the control point. The duplication is *defence in depth*
between two components with different trust levels, not two copies of one
check inside one component.

The bound is declared once in `policies/policy.yaml` and the twin reads
its own from `config/plant.yaml`; they are separately owned on purpose.

## Why a test rather than a repository split
A separate repository enforces nothing: one `pip install cooling-twin`
and the boundary is gone with no signal. Three checks run instead —
an AST scan of gateway sources, the declared dependency list, and a
subprocess `sys.modules` probe for transitive pull-in. The first two need
nothing installed, so the rule holds on a bare checkout.

`tests/fakes/fake_twin.py` is the positive demonstration: the gateway
suite passes with `cooling-twin` uninstalled.

## Alternatives
**Separate repositories.** Considered and rejected — see above. It also
loses the single portfolio narrative (physics → calibration → G14 gate →
policy gateway) and adds a version-pin problem where a git tag suffices.

**import-linter or tach.** Would express the rule declaratively. Deferred:
one more dependency and one more config file to express a rule that is
~40 lines of AST, and the hand-written version can carry its own
"guards the guard" test.

## Consequences
- Anything the gateway needs from the twin must become an MCP tool
- Adding `cooling-twin` to the gateway's dependencies requires changing
  this ADR, not the test
- The gateway suite runs fast and offline, with no BDG2 and no ODE solve

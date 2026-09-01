"""MCP policy gateway for building cooling control.

Sits between an AI agent and a cooling digital twin. Authenticates the
caller, classifies every tool into a capability tier, and enforces the
rule that no setpoint reaches the twin unless the same principal has
already simulated that exact change and holds an unexpired, single-use
receipt proving it.

SCAFFOLD ONLY (M0). No policy, auth or receipt logic lives here yet --
see docs/MILESTONES.md for what lands in which milestone, and
docs/adr/ for the decisions already taken. The one thing this package
enforces from its first commit is what it must NOT contain: it does not
import `cooling_twin` or `twin_mcp`, and `tests/test_boundary.py` fails
the build if that ever changes.

Module layout as milestones land:

    auth.py       M4  JWT verification -> Principal
    policy.py     M5  policy.yaml v2 evaluation, tiers, bounds, caps
    receipts.py   M6  minting and the seven validation checks -- and the
                      ONLY place receipt verification may live
    audit.py      M7  one record per request, denials included
    proxy.py      M3  MCP transport to the upstream twin
    store.py      M2  Redis-backed state: simulation cache, consumed
                      nonces, per-zone last-applied setpoint
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

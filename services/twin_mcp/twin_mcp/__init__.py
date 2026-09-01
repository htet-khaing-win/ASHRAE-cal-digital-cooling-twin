"""MCP server exposing the cooling digital twin's tools.

THE ADAPTER, and the only package permitted to import `cooling_twin`.
Physics on one side, an MCP tool contract on the other. The gateway
talks to this over MCP and imports neither it nor the twin -- see
services/gateway/tests/test_boundary.py.

SCAFFOLD ONLY (M0). Tools land in M1 (read) and M2 (simulate); the
apply tool writes to SIMULATED control state, since no BMS is
connected to this project (docs/ARCHITECTURE.md).

Every schema field carries its unit in its name -- `setpoint_c`,
`horizon_hours`, `window_minutes` -- never a bare `temperature` or
`duration`. That convention is T-13's mitigation and is enforced by
review, not by a type.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

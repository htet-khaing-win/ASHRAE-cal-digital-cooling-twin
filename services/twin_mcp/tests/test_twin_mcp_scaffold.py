"""The adapter package exists and is the one place allowed to reach the twin.

M0 ships no tools; M1 adds the read tools. What this file pins now is
the half of ADR-0007 that applies on THIS side of the boundary. The
gateway's own suite proves it does not import the twin; nothing there
proves the adapter is permitted to, and an inverted rule would be a
silent architectural change.
"""

from __future__ import annotations

import twin_mcp


def test_twin_mcp_package_imports_and_declares_a_version() -> None:
    """M0 has no tools yet -- assert only what is true."""
    assert twin_mcp.__version__
    assert twin_mcp.__all__ == ["__version__"]


def test_the_adapter_is_the_package_allowed_to_depend_on_the_twin() -> None:
    """`cooling-twin` is declared here and banned in the gateway.

    Read from the manifests rather than asserted in prose, so the two
    sides of the boundary cannot drift apart without a test failing.
    """
    import tomllib
    from pathlib import Path

    adapter_root = Path(__file__).resolve().parent.parent
    manifest = tomllib.loads((adapter_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = manifest["project"]["dependencies"]
    assert any(requirement.startswith("cooling-twin") for requirement in dependencies), (
        "services/twin_mcp is the adapter: it is the ONLY package permitted to "
        "depend on the twin, and removing that dependency would leave nothing "
        "able to reach the physics (ADR-0007)."
    )

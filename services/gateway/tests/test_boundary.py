"""The gateway must not depend on the twin. This file is that rule, executable.

WHY THIS IS A TEST AND NOT A CONVENTION. The project's central claim is
that the gateway is a general control point which happens to sit in
front of a cooling twin -- "the twin may be replaced; the gateway is the
control point" (docs/ARCHITECTURE.md). That claim is worth exactly as
much as the mechanism enforcing it. A repository split would not enforce
it: one `pip install cooling-twin` and one `import cooling_twin` and the
claim is quietly false with nobody notified. A test fails loudly, in CI,
on the commit that breaks it.

It lands in the scaffold, BEFORE any gateway logic exists, so it can
never be retrofitted around a violation that is already shipping.

THREE INDEPENDENT CHECKS, because each catches what the others miss:

  1. AST scan of gateway sources     -- catches a direct `import cooling_twin`
  2. Declared dependencies           -- catches `cooling-twin` added to pyproject
  3. Runtime `sys.modules` probe     -- catches a TRANSITIVE pull-in through
                                        some other dependency (skipped when the
                                        gateway's own deps are not installed)

Checks 1 and 2 are pure text/AST work and run with nothing installed, so
the boundary is enforced from the first commit of the scaffold onward.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

GATEWAY_ROOT = Path(__file__).resolve().parent.parent
GATEWAY_PACKAGE = GATEWAY_ROOT / "gateway"
GATEWAY_PYPROJECT = GATEWAY_ROOT / "pyproject.toml"

# The twin, in both the spellings that would appear -- the import name
# and the distribution name. `twin_mcp` is banned for the same reason:
# it is the adapter that DOES import the twin, so depending on it would
# reach `cooling_twin` one hop later and defeat the whole arrangement.
FORBIDDEN_IMPORT_ROOTS = frozenset({"cooling_twin", "twin_mcp"})
# Normalised PEP 503 form (lowercase, hyphens) -- `_requirement_name`
# reduces every spelling to this before comparing.
FORBIDDEN_DISTRIBUTIONS = frozenset({"cooling-twin", "twin-mcp"})

REMEDY = (
    "The gateway reaches the twin over MCP, never by import. If you need "
    "data from the twin, call the tool; if the tool does not exist yet, add "
    "it to services/twin_mcp/. If you believe the gateway genuinely needs to "
    "import the twin, that is a change to docs/adr/0007-gateway-twin-boundary.md, "
    "not a change to this test."
)


def _gateway_sources() -> list[Path]:
    """Every Python file in the gateway package."""
    return sorted(GATEWAY_PACKAGE.rglob("*.py"))


def _imported_roots(source: Path) -> set[str]:
    """Top-level module names imported by one file.

    Parsed rather than grepped: a grep matches the word inside a comment,
    a docstring, or a string literal naming the twin in prose -- all of
    which are legitimate and none of which are dependencies. This file's
    own module docstring says `cooling_twin` several times and must not
    trip its own check.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        # `node.module` is None for a relative import (`from . import x`),
        # which is inside the gateway package by definition and fine.
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_gateway_source_imports_the_twin() -> None:
    """Check 1: no direct import, anywhere under `gateway/`."""
    offenders: list[str] = []
    for source in _gateway_sources():
        forbidden = _imported_roots(source) & FORBIDDEN_IMPORT_ROOTS
        offenders.extend(
            f"{source.relative_to(GATEWAY_ROOT)} imports {name}" for name in sorted(forbidden)
        )
    assert not offenders, (
        "gateway must not import the twin:\n  " + "\n  ".join(offenders) + f"\n\n{REMEDY}"
    )


def _requirement_name(requirement: str) -> str:
    """The distribution name from a PEP 508 requirement string, normalised.

    `cooling-twin`, `cooling_twin>=1.0`, `cooling-twin[dev]==2`, and
    `cooling-twin ; python_version<"3.12"` must all reduce to the same
    token. PEP 503 treats runs of `-`, `_` and `.` as equivalent and
    names as case-insensitive, so normalise to lowercase hyphens rather
    than trusting the spelling someone happens to use.
    """
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if match is None:
        return ""
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def test_gateway_does_not_declare_the_twin_as_a_dependency() -> None:
    """Check 2: the twin is absent from the declared dependency list.

    Catches the violation one step earlier than an import does -- a
    dependency declared but not yet used is a boundary already breached
    in intent, and it is far cheaper to remove before code depends on it.
    """
    manifest = tomllib.loads(GATEWAY_PYPROJECT.read_text(encoding="utf-8"))
    project = manifest["project"]
    declared = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)

    offenders = [
        requirement
        for requirement in declared
        if _requirement_name(requirement) in FORBIDDEN_DISTRIBUTIONS
    ]
    assert not offenders, (
        f"services/gateway/pyproject.toml declares {offenders}, which breaks the "
        f"gateway/twin boundary.\n\n{REMEDY}"
    )


def test_importing_the_gateway_does_not_pull_in_the_twin() -> None:
    """Check 3: nothing transitively drags the twin into the process.

    Runs in a SUBPROCESS with a clean interpreter. Doing it in-process
    would be meaningless: the twin is installed in this environment and
    the rest of the suite may already have imported it, so `sys.modules`
    here says nothing about what importing the gateway costs.

    Skipped when the gateway package is not importable (its own
    dependencies are not installed in the twin's conda env by default),
    because a missing `mcp` must not masquerade as a boundary violation.
    """
    probe = (
        "import sys, importlib;"
        "importlib.import_module('gateway');"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in "
        f"{sorted(FORBIDDEN_IMPORT_ROOTS)!r});"
        "print('LEAKED:' + ','.join(leaked))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=GATEWAY_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        detail = stderr.splitlines()[-1] if stderr else "unknown"
        pytest.skip(
            "gateway package not importable in this environment "
            f"(install `-e ./services/gateway` to enable this check): {detail}"
        )
    leaked = completed.stdout.split("LEAKED:")[-1].strip()
    assert not leaked, (
        f"importing `gateway` transitively imported {leaked}. Something in the "
        f"dependency graph reaches the twin.\n\n{REMEDY}"
    )


def test_the_check_can_actually_fail(tmp_path: Path) -> None:
    """Guards the guard.

    Checks 1 and 2 pass trivially against an empty package, and an empty
    package is exactly what the scaffold ships. Prove the AST scan
    detects a violation before trusting it to have found none.
    """
    planted = tmp_path / "violation.py"
    planted.write_text("import cooling_twin\nfrom twin_mcp.server import x\n", encoding="utf-8")
    assert _imported_roots(planted) & FORBIDDEN_IMPORT_ROOTS == {"cooling_twin", "twin_mcp"}

    for spelling in (
        "cooling-twin",
        "cooling_twin>=1.0",
        "cooling-twin[dev]==2.0",
        'cooling-twin ; python_version<"3.12"',
        "  COOLING.TWIN  ",
        "twin_mcp",
    ):
        assert _requirement_name(spelling) in FORBIDDEN_DISTRIBUTIONS, spelling
    for allowed in ("mcp", "pydantic", "pyjwt[crypto]", "httpx>=0.27"):
        assert _requirement_name(allowed) not in FORBIDDEN_DISTRIBUTIONS, allowed

    innocent = tmp_path / "innocent.py"
    innocent.write_text(
        '"""Talks to cooling_twin over MCP."""\n'
        "# cooling_twin is never imported here\n"
        "NAME = 'cooling_twin'\n"
        "from . import sibling\n",
        encoding="utf-8",
    )
    assert not _imported_roots(innocent) & FORBIDDEN_IMPORT_ROOTS


def test_gateway_package_exists_and_is_scanned() -> None:
    """A boundary test that scans nothing passes for the wrong reason."""
    assert GATEWAY_PACKAGE.is_dir(), f"{GATEWAY_PACKAGE} missing"
    assert _gateway_sources(), "no gateway sources found -- the scan covered zero files"

"""Import paths for the gateway suite, and the reason they are shaped this way.

Two insertions, for two different reasons.

`tests/` -- ALWAYS. It makes the test doubles importable as `fakes.*`,
a name that is unique across the whole repository. The obvious
alternative, making `tests/` a package and importing `tests.fakes`,
collides head-on with the repository root's own `tests/` package: with
both on sys.path, `import tests` resolves to whichever came first and a
bare `pytest` at the repo root fails to collect the twin's 680 tests.
Explicit paths in the Makefile hid that; a developer typing `pytest`
would not have been so lucky.

`services/gateway` -- ONLY when the `gateway` package is not already
importable. `environment.yml` installs it editable, but the suite must
also run on a bare checkout: 05_ENGINEERING_STANDARDS.md is explicit
that nothing may block work, and a boundary test that only runs after a
successful `pip install` protects nothing on the commit that breaks it.
An installed copy always wins, so the two never shadow each other.

This lives here rather than in `services/` because this directory holds
a `pyproject.toml`, which makes it pytest's rootdir -- a conftest one
level up sits above the rootdir and is never loaded.

NOTE neither insertion puts the repository's `src/` on the path. The
gateway must not be able to import `cooling_twin` even by accident
during tests -- see tests/test_boundary.py and ADR-0007.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PACKAGE_ROOT / "tests"))

if importlib.util.find_spec("gateway") is None:
    sys.path.insert(0, str(PACKAGE_ROOT))

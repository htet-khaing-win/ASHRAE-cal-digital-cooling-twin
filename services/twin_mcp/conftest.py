"""Make `twin_mcp` importable without installing it.

`environment.yml` installs this package editable, but the suite must
also run on a bare checkout -- 05_ENGINEERING_STANDARDS.md is explicit
that nothing may block work, and a boundary test that only runs after a
successful `pip install` protects nothing on the commit that breaks it.

Same trick `dashboard/data.py` uses to reach `scripts/`: one sys.path
insertion at collection time, applied only when the package is not
already importable, so an installed copy always wins over the checkout.

It lives HERE rather than in `services/` because this directory holds a
`pyproject.toml`, which makes it pytest's rootdir -- a conftest one level
up sits above the rootdir and is never loaded.

NOTE this deliberately does NOT put the repository's `src/` on the path.
This package reaches the twin through its declared `cooling-twin`
dependency, which is an install, not a path hack.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

if importlib.util.find_spec("twin_mcp") is None:
    sys.path.insert(0, str(PACKAGE_ROOT))

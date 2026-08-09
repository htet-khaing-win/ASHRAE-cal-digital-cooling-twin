#!/usr/bin/env python3
"""Environment verification for the cooling-twin project.
"""

from __future__ import annotations

import platform
import sys
from importlib import import_module
from pathlib import Path

REQUIRED_PYTHON = (3, 11)
REQUIRED_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "psychrolib")
DATA_DIR = Path.home() / "data" / "bdg2"


def check_python_version() -> None:
    """Verify the running interpreter is Python 3.11.x.

    Raises:
        SystemExit: if the interpreter's major.minor version does not
            match REQUIRED_PYTHON.
    """
    current = sys.version_info[:2]
    if current != REQUIRED_PYTHON:
        print(
            f"[FAIL] Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} required, "
            f"found {current[0]}.{current[1]}.\n"
            f"       Fix: conda activate cooling-twin"
        )
        sys.exit(1)
    print(f"[OK]   Python {current[0]}.{current[1]}")


def check_imports() -> None:
    """Verify every required scientific package imports cleanly.

    Checks packages one at a time so a failure names exactly which
    package is missing, rather than a single opaque ImportError.

    Raises:
        SystemExit: on the first import failure.
    """
    for name in REQUIRED_PACKAGES:
        try:
            import_module(name)
        except ImportError as exc:
            print(
                f"[FAIL] Cannot import '{name}': {exc}\n"
                f"       Fix: conda env update -f environment.yml"
            )
            sys.exit(1)
        print(f"[OK]   import {name}")


def check_filesystem_location() -> None:
    """Verify the interpreter is running under WSL2, and that the project
    data directory does not resolve onto the /mnt/c Windows mount.

    Raises:
        SystemExit: if running on native Windows, or if DATA_DIR resolves
            under /mnt/c.
    """
    system = platform.system()
    release = platform.release().lower()
    is_wsl = "microsoft" in release or "wsl" in release

    if system == "Windows":
        print(
            "[FAIL] Running on native Windows Python, not WSL2.\n"
            "       This project requires WSL2 Ubuntu 24.04. Native Windows\n"
            "       Python does not share WSL2's filesystem performance or\n"
            "       POSIX semantics that several packages assume."
        )
        sys.exit(1)
    print("[OK]   Running inside WSL2" if is_wsl else "[OK]   Running on native Linux")

    resolved = DATA_DIR.expanduser().resolve() if DATA_DIR.exists() else DATA_DIR
    if str(resolved).startswith("/mnt/c"):
        print(
            f"[FAIL] Data directory resolves to {resolved}, under /mnt/c.\n"
            f"       Cross-boundary I/O is 10-20x slower than native WSL2\n"
            f"       disk. Move the data to ~/data/bdg2 instead."
        )
        sys.exit(1)
    print(f"[OK]   Data path target: {DATA_DIR} (not /mnt/c)")


def main() -> None:
    """Run every check in order, stopping at the first failure."""
    print("=== cooling-twin environment check ===")
    check_python_version()
    check_imports()
    check_filesystem_location()
    print("=== all checks passed ===")


if __name__ == "__main__":
    main()
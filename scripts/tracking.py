"""MLflow logging for the tuning experiments -- optional, never required.

WHY THIS LIVES IN `scripts/` AND NOT IN `src/cooling_twin/`. The same
rule that put `optuna` and `SALib` in the dev dependencies: nothing
importable from `src/` may depend on an experiment-tracking service. A
deployment install of this package must not pull in MLflow, Flask,
SQLAlchemy and a web server so that a thermal model can predict a
cooling load. Tracking is something the ANALYST does around the model,
not something the model does.

WHY EVERY CALL HERE NO-OPS WHEN MLFLOW IS ABSENT. The alternative is
`if mlflow_enabled:` at thirty call sites, which is how tracking code
becomes the thing that breaks a run. The scripts call these helpers
unconditionally; if MLflow is not installed, or `--mlflow` was not
passed, the helpers do nothing and the analysis produces exactly the
same JSON artifacts it always did. The JSON artifacts remain the source
of truth -- MLflow is for COMPARING runs, not for recording them, and a
result that exists only inside `mlruns/` is a result that cannot be
reviewed in a pull request.

Default store is a local SQLite file, `sqlite:///mlflow.db`, with
artifacts alongside it in `mlartifacts/`. No server to keep running.
Browse it with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

NOT the `./mlruns` directory store that most MLflow tutorials open with:
MLflow 3.x puts that backend in maintenance mode and raises on startup
unless `MLFLOW_ALLOW_FILE_STORE=true` is set. SQLite is the supported
path, is still a single file needing no server, and -- unlike the
directory store -- supports the run comparison this is being set up for.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger("tracking")

# Local SQLite store, relative to the repo root. Overridable through the
# standard MLflow environment variable so a shared server can be pointed
# at without touching this file.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

# One experiment for everything in M7's tuning work. Runs are separated
# by name and tags, not by experiment -- comparing across experiments in
# the MLflow UI is deliberately awkward, and the whole point here is
# comparison.
DEFAULT_EXPERIMENT = "cooling-twin-hybrid-tuning"

_ACTIVE = False


def _mlflow() -> Any | None:
    """Import MLflow, or return None if it is not installed."""
    if not _ACTIVE:
        return None
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "tracking was requested but mlflow is not installed; continuing "
            "without it. `pip install mlflow`, or drop --mlflow."
        )
        return None
    return mlflow


def enable(experiment: str = DEFAULT_EXPERIMENT, uri: str | None = None) -> bool:
    """Turn tracking on for this process.

    Args:
        experiment: MLflow experiment name.
        uri: Tracking URI. Defaults to `MLFLOW_TRACKING_URI` if set,
            otherwise the local file store.

    Returns:
        Whether tracking is actually active. False means MLflow is not
        installed and every helper below will no-op.
    """
    global _ACTIVE
    _ACTIVE = True
    mlflow = _mlflow()
    if mlflow is None:
        _ACTIVE = False
        return False

    resolved = uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(resolved)
    mlflow.set_experiment(experiment)
    logger.info("MLflow tracking to %s, experiment %r", resolved, experiment)
    return True


def git_commit() -> str:
    """The current commit, or `"unknown"` outside a git checkout.

    Logged as a tag on every run. A tuning result whose code version is
    not recorded cannot be reproduced, and this project's search space,
    fold layout and objective all live in code rather than in config.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip() or "unknown"


@contextmanager
def run(name: str, tags: Mapping[str, Any] | None = None) -> Iterator[None]:
    """A tracked run, or a no-op context if tracking is off.

    Args:
        name: Run name, shown in the MLflow UI.
        tags: Extra tags. The git commit is added automatically.

    Yields:
        Nothing. Use `log_params` / `log_metrics` inside the block.
    """
    mlflow = _mlflow()
    if mlflow is None:
        yield
        return
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"git_commit": git_commit(), **dict(tags or {})})
        yield


def log_params(params: Mapping[str, Any]) -> None:
    """Log run parameters, if tracking is on."""
    mlflow = _mlflow()
    if mlflow is not None:
        mlflow.log_params(dict(params))


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log run metrics, if tracking is on.

    Non-finite values are dropped rather than logged. MLflow stores NaN
    happily and then renders it as an empty cell, which in a comparison
    table is indistinguishable from "this run did not report it".
    """
    mlflow = _mlflow()
    if mlflow is None:
        return
    clean = {
        key: float(value)
        for key, value in metrics.items()
        if value == value and abs(float(value)) != float("inf")
    }
    dropped = set(metrics) - set(clean)
    if dropped:
        logger.warning("not logging non-finite metrics: %s", sorted(dropped))
    mlflow.log_metrics(clean, step=step)


def log_json(payload: Any, filename: str) -> None:
    """Attach a JSON artifact to the active run, if tracking is on."""
    mlflow = _mlflow()
    if mlflow is not None:
        mlflow.log_dict(payload, filename)


def log_file(path: Path) -> None:
    """Attach an existing file to the active run, if tracking is on."""
    mlflow = _mlflow()
    if mlflow is not None and path.exists():
        mlflow.log_artifact(str(path))

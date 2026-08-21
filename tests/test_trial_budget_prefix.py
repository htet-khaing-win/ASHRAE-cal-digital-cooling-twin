"""The assumption `sweep_trial_budget.py` rests on, asserted (L7.3).

`scripts/sweep_trial_budget.py` runs ONE Optuna study at the largest
budget and reads the running best at 25, 50, 75, ... trials, instead of
running six separate studies. That is a 2.5x saving, and it is only
valid if a study's first `k` trials are identical to the whole of a
`k`-trial study.

They are, because `TPESampler(seed=...)` is a sequential deterministic
process: trial `i` depends on the seed and on trials `0..i-1`, nothing
else. But "should be deterministic" is exactly the kind of assumption
that quietly stops holding across a library version, and the saving is
worthless if it silently changes the answer. So it is tested rather than
assumed -- and the test is written against the SAMPLER, on a cheap
closed-form objective, not against the residual model, so it runs in
milliseconds and fails for the right reason.

This is the first test in the project that targets `scripts/` rather
than `src/`. It earns the exception because it is not testing a script's
orchestration -- it is testing a mathematical claim that a script's
correctness depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from cooling_twin import SEED

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

optuna = pytest.importorskip("optuna", reason="optuna is a dev dependency")


def _study(n_trials: int) -> optuna.Study:
    """Run a study on a deterministic objective with the project seed.

    The objective is a fixed six-dimensional function chosen to have the
    same SHAPE as the real search space -- integer, float and log-scaled
    parameters over the same ranges as `tune_residual_model._suggest` --
    so the sampler exercises the same code paths it does in anger.
    """

    def objective(trial: optuna.Trial) -> float:
        depth = trial.suggest_int("max_depth", 2, 8)
        leaves = trial.suggest_int("max_leaf_nodes", 7, 63, log=True)
        leaf = trial.suggest_int("min_samples_leaf", 10, 200, log=True)
        rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        iterations = trial.suggest_int("max_iter", 100, 800, step=100)
        l2 = trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True)
        return (
            (depth - 4) ** 2
            + (leaves - 15) ** 2 / 100.0
            + (leaf - 50) ** 2 / 1000.0
            + (rate - 0.05) ** 2 * 1000.0
            + (iterations - 400) ** 2 / 1e5
            + l2
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def test_a_long_study_contains_the_short_studies_trial_for_trial() -> None:
    """The first k trials of a long study are a short study's k trials."""
    short, long = _study(12), _study(30)

    assert len(short.trials) == 12
    for index, (a, b) in enumerate(zip(short.trials, long.trials[:12], strict=True)):
        assert a.params == b.params, f"trial {index} diverged"
        assert a.value == pytest.approx(b.value), f"trial {index} scored differently"


def test_the_running_best_matches_the_short_studys_best() -> None:
    """What the sweep actually reads off, checked against the real thing.

    `sweep_trial_budget` takes `min(study.trials[:k])`. That must equal
    the `best_params` a k-trial study reports -- if it did not, every
    row of the sweep below the largest budget would be fiction.
    """
    short, long = _study(12), _study(30)

    running_best = min(long.trials[:12], key=lambda trial: trial.value)
    assert running_best.params == short.best_params
    assert running_best.value == pytest.approx(short.best_value)


def test_the_budget_sweep_refuses_a_checkpoint_it_cannot_honour() -> None:
    """Asking for 300 trials of a 100-trial study is a bug, not a clamp.

    Returning the 100-trial answer under a "300" key would produce a
    sweep table whose last two rows were identical for a reason nothing
    in the output explained.
    """
    import numpy as np
    import pandas as pd
    from tune_residual_model import tune

    from cooling_twin.analysis.hybrid import build_features

    index = pd.date_range("2016-01-01", periods=4000, freq="h")
    features = build_features(index, np.full(4000, 20.0), np.full(4000, 0.008))

    with pytest.raises(ValueError, match="between 1 and n_trials"):
        tune(
            features,
            np.zeros(4000),
            n_trials=5,
            n_inner_folds=2,
            label="refusal",
            checkpoints=(5, 300),
        )

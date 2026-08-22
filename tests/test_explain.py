"""Tests for exact Shapley attribution and the comparison it feeds (L7.4).

Same two-layer split as `test_hybrid.py`, for the same reason.

The known-answer layer uses models whose Shapley values can be written
down: for an additive linear model the attribution of feature j is
exactly `coefficient_j * (x_j - mean(background_j))`, and for a product
of two symmetric features it is exactly half the prediction each. A
failure in those tests is a failure of the coalition arithmetic, not of
a tree ensemble's internals.

The second layer fits the real ensemble, and the sharpest test in the
file is `test_attribution_ranks_a_model_that_learnt_nothing`: it fits
the booster on a SHUFFLED target, where the true answer is that no
feature matters, and asserts that the attribution nevertheless produces
a confident ranking while the held-out permutation cost does not. That
test is the lesson. If it ever fails -- if attribution starts collapsing
to zero on a worthless model -- the argument in this module's docstring
is wrong and the report needs rewriting, not the test.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from cooling_twin import set_seed
from cooling_twin.analysis import explain as explain_module
from cooling_twin.analysis.explain import (
    MAX_EXACT_FEATURES,
    ShapleyExplanation,
    explanation_comparison,
    rank_agreement,
    sample_rows,
    shapley_values_kw,
)
from cooling_twin.analysis.hybrid import (
    build_features,
    default_model_factory,
    permutation_importance_kw,
)
from cooling_twin.calibration.crossval import expanding_window_folds


class _LinearModel:
    """f(x) = x @ coefficients. Shapley values are known in closed form."""

    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = np.asarray(coefficients, dtype=float)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) @ self.coefficients


class _ProductModel:
    """f(x) = x0 * x1. Pure interaction, no main effect at the origin."""

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        return values[:, 0] * values[:, 1]


class _NoisyModel:
    """A model whose predict() is not a function of its input."""

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.rng.normal(0.0, 100.0, size=len(x))


def _frame(**columns: object) -> pd.DataFrame:
    return pd.DataFrame({name: np.asarray(values, dtype=float) for name, values in columns.items()})


def _synthetic_residual_year(
    hours: int = 8760,
) -> tuple[pd.DataFrame, np.ndarray]:
    """A feature frame and a residual carrying two known switched terms."""
    rng = set_seed()
    index = pd.date_range("2016-01-01", periods=hours, freq="h")
    steps = np.arange(hours, dtype=float)
    t_outdoor = (
        20.0
        + 12.0 * np.sin(2.0 * np.pi * (steps - 2000.0) / 8760.0)
        + 5.0 * np.sin(2.0 * np.pi * steps / 24.0)
    )
    humidity = 0.006 + 0.004 * np.sin(2.0 * np.pi * (steps - 2000.0) / 8760.0)
    residual = (
        40.0 * np.maximum(t_outdoor - 28.0, 0.0)
        + 25.0 * np.maximum(humidity * 1000.0 - 8.0, 0.0)
        + rng.normal(0.0, 90.0, size=hours)
    )
    return build_features(index, t_outdoor, humidity), residual


# --------------------------------------------------------------------
# known answers -- the coalition arithmetic
# --------------------------------------------------------------------


def test_linear_model_attribution_is_coefficient_times_centred_feature() -> None:
    """The one case where the exact answer is on paper."""
    background = _frame(a=[0.0, 2.0, 4.0, 6.0], b=[10.0, 10.0, 20.0, 20.0])
    explain = _frame(a=[5.0, 1.0], b=[30.0, 0.0])
    model = _LinearModel(np.array([2.0, 3.0]))

    result = shapley_values_kw(model, explain, background, plausibility_pairs=())

    # mean background: a = 3.0, b = 15.0
    expected = np.array(
        [
            [2.0 * (5.0 - 3.0), 3.0 * (30.0 - 15.0)],
            [2.0 * (1.0 - 3.0), 3.0 * (0.0 - 15.0)],
        ]
    )
    assert result.values_kw == pytest.approx(expected)
    assert result.base_value_kw == pytest.approx(2.0 * 3.0 + 3.0 * 15.0)
    assert result.prediction_kw == pytest.approx([2.0 * 5.0 + 3.0 * 30.0, 2.0 * 1.0 + 3.0 * 0.0])


def test_attributions_sum_to_prediction_minus_base_value() -> None:
    """Efficiency, on the real ensemble rather than a stub."""
    features, residual = _synthetic_residual_year(hours=2000)
    model = default_model_factory()()
    model.fit(features.to_numpy(dtype=float), residual)

    result = shapley_values_kw(
        model, sample_rows(features, 20), sample_rows(features, 40, seed=1)
    )

    assert result.values_kw.sum(axis=1) == pytest.approx(
        result.prediction_kw - result.base_value_kw, abs=1e-9
    )
    assert result.efficiency_error_kw < 1e-9


def test_a_feature_the_model_ignores_scores_exactly_zero() -> None:
    """The dummy axiom, tested where it is least obvious: under correlation.

    `b` is a perfect copy of `a`, so any correlational account would
    happily credit it. The interventional Shapley value gives it zero,
    because the model never reads that column -- which is precisely why
    these values describe the MODEL and not the building.
    """
    values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    background = _frame(a=values, b=values)
    explain = _frame(a=[4.0], b=[4.0])
    model = _LinearModel(np.array([7.0, 0.0]))

    result = shapley_values_kw(model, explain, background, plausibility_pairs=())

    assert result.values_kw[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert result.values_kw[0, 0] == pytest.approx(7.0 * (4.0 - 2.5))


def test_symmetric_features_split_a_pure_interaction_evenly() -> None:
    """f = a*b, background at the origin: half the prediction each."""
    background = _frame(a=[0.0, 0.0, 0.0], b=[0.0, 0.0, 0.0])
    explain = _frame(a=[4.0], b=[5.0])

    result = shapley_values_kw(_ProductModel(), explain, background, plausibility_pairs=())

    assert result.base_value_kw == pytest.approx(0.0)
    assert result.values_kw[0] == pytest.approx([10.0, 10.0])


def test_attribution_is_deterministic_under_the_project_seed() -> None:
    """Two identical calls give bit-identical attributions."""
    features, residual = _synthetic_residual_year(hours=1500)
    model = default_model_factory()()
    model.fit(features.to_numpy(dtype=float), residual)
    explain, background = sample_rows(features, 15), sample_rows(features, 40, seed=1)

    first = shapley_values_kw(model, explain, background)
    second = shapley_values_kw(model, explain, background)

    assert np.array_equal(first.values_kw, second.values_kw)


# --------------------------------------------------------------------
# the fabricated-input diagnostic
# --------------------------------------------------------------------


def test_locked_pair_reports_the_exact_fabrication_rate() -> None:
    """A pair that is always equal is fabricated in exactly half the rows.

    Half the coalitions take one of the two columns from the explained
    row and the other from a background row; here every one of those
    fabricates a difference the data never shows. The closed-form count
    in `_off_manifold_fraction` must return exactly 0.5.
    """
    background = _frame(a=[0.0, 1.0, 2.0, 3.0], b=[0.0, 1.0, 2.0, 3.0])
    explain = _frame(a=[10.0, 11.0], b=[10.0, 11.0])

    result = shapley_values_kw(
        _LinearModel(np.array([1.0, 1.0])),
        explain,
        background,
        plausibility_pairs=(("a", "b"),),
    )

    assert result.off_manifold_fraction["a vs b"] == pytest.approx(0.5)
    assert result.worst_off_manifold_fraction == pytest.approx(0.5)


def test_an_unconstrained_pair_fabricates_nothing() -> None:
    """When every combination is already observed, the rate is zero."""
    background = _frame(a=[0.0, 10.0, 0.0, 10.0], b=[0.0, 0.0, 10.0, 10.0])
    explain = _frame(a=[5.0], b=[5.0])

    result = shapley_values_kw(
        _LinearModel(np.array([1.0, 1.0])),
        explain,
        background,
        plausibility_pairs=(("a", "b"),),
    )

    assert result.off_manifold_fraction["a vs b"] == pytest.approx(0.0)


def test_default_pairs_are_measured_on_the_project_feature_frame() -> None:
    """Both declared pairs are picked up automatically, and both bind."""
    features, _ = _synthetic_residual_year(hours=1500)
    model = _LinearModel(np.ones(len(features.columns)))

    result = shapley_values_kw(model, sample_rows(features, 10), sample_rows(features, 40, seed=1))

    assert set(result.off_manifold_fraction) == {
        "outdoor_dry_bulb_c vs outdoor_dry_bulb_24h_mean_c",
        "hour_sin vs hour_cos",
    }
    assert result.worst_off_manifold_fraction > 0.0


# --------------------------------------------------------------------
# rejections
# --------------------------------------------------------------------


def test_a_wrong_coalition_weight_is_caught_by_the_efficiency_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The permanent meta-test: prove the self-check has teeth.

    A single perturbed factorial is the smallest realistic version of
    the bug this implementation is exposed to -- an off-by-one in
    `|S|! (k - |S| - 1)! / k!`. The values it produces still look like
    attributions; only the sum gives it away.
    """
    background = _frame(a=[0.0, 1.0, 2.0, 3.0], b=[3.0, 1.0, 4.0, 0.0])
    explain = _frame(a=[1.0], b=[2.0])
    real_factorial = explain_module.factorial
    monkeypatch.setattr(
        explain_module, "factorial", lambda n: real_factorial(n) + (1 if n == 1 else 0)
    )

    with pytest.raises(ValueError, match="coalition weight"):
        shapley_values_kw(
            _LinearModel(np.array([2.0, 3.0])), explain, background, plausibility_pairs=()
        )


def test_a_non_deterministic_model_is_not_caught_and_that_is_documented() -> None:
    """Efficiency is an identity over the coalition table, not a validity check.

    Asserted rather than left implicit, because the natural assumption
    -- that a self-check which raises must be checking the model -- is
    wrong, and acting on it would mean shipping attributions from a
    model whose predictions are noise. Only the held-out columns of
    `explanation_comparison` speak to that.
    """
    background = _frame(a=[0.0, 1.0, 2.0, 3.0], b=[0.0, 1.0, 2.0, 3.0])
    explain = _frame(a=[1.0], b=[2.0])

    result = shapley_values_kw(_NoisyModel(), explain, background, plausibility_pairs=())

    assert result.efficiency_error_kw < 1e-9


@pytest.mark.parametrize(
    ("explain", "background", "message"),
    [
        (_frame(a=[1.0], b=[1.0]), _frame(b=[1.0], a=[1.0]), "same columns in the same order"),
        (_frame(a=[], b=[]), _frame(a=[1.0], b=[1.0]), "non-empty"),
        (_frame(a=[1.0]), _frame(a=[1.0]), "attribution over one feature"),
    ],
)
def test_bad_frames_are_rejected(
    explain: pd.DataFrame, background: pd.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        shapley_values_kw(_LinearModel(np.ones(len(explain.columns))), explain, background)


def test_too_many_features_is_refused_rather_than_approximated() -> None:
    """2^k is the whole cost model; the limit is stated, not discovered."""
    columns = {f"f{i}": [1.0] for i in range(MAX_EXACT_FEATURES + 1)}
    frame = _frame(**columns)

    with pytest.raises(ValueError, match="above the exact limit"):
        shapley_values_kw(_LinearModel(np.ones(len(frame.columns))), frame, frame)


def test_the_evaluation_budget_is_checked_before_any_prediction() -> None:
    """A careless call fails in milliseconds, not in an hour."""
    frame = _frame(**{f"f{i}": np.ones(1000) for i in range(MAX_EXACT_FEATURES)})

    with pytest.raises(ValueError, match="above the ceiling"):
        shapley_values_kw(_LinearModel(np.ones(MAX_EXACT_FEATURES)), frame, frame)


def test_an_absent_plausibility_column_is_rejected() -> None:
    frame = _frame(a=[1.0, 2.0], b=[1.0, 2.0])

    with pytest.raises(ValueError, match="absent column"):
        shapley_values_kw(
            _LinearModel(np.ones(2)), frame, frame, plausibility_pairs=(("a", "missing"),)
        )


def test_a_thin_background_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    background = _frame(a=[0.0, 1.0], b=[0.0, 1.0])
    explain = _frame(a=[1.0], b=[1.0])

    with caplog.at_level(logging.WARNING):
        shapley_values_kw(
            _LinearModel(np.ones(2)), explain, background, plausibility_pairs=(), label="thin"
        )

    assert "lottery" in caplog.text


# --------------------------------------------------------------------
# sampling and the comparison table
# --------------------------------------------------------------------


def test_sample_rows_keeps_input_order_and_is_reproducible() -> None:
    frame = _frame(a=np.arange(100.0))

    first = sample_rows(frame, 10)
    second = sample_rows(frame, 10)

    assert list(first.index) == sorted(first.index)
    assert first.equals(second)
    assert len(first) == 10


def test_sample_rows_returns_everything_when_asked_for_more_than_it_has() -> None:
    frame = _frame(a=np.arange(5.0))

    assert sample_rows(frame, 50).equals(frame)


@pytest.mark.parametrize(
    ("frame", "n_rows", "message"),
    [
        (_frame(a=np.arange(5.0)), 0, "must be positive"),
        (_frame(a=[]), 3, "empty"),
    ],
)
def test_sample_rows_rejects_bad_input(frame: pd.DataFrame, n_rows: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_rows(frame, n_rows)


def _explanation(**mean_abs: float) -> ShapleyExplanation:
    """A hand-built explanation whose mean |attribution| is chosen."""
    names = tuple(mean_abs)
    return ShapleyExplanation(
        label="stub",
        feature_names=names,
        values_kw=np.array([[mean_abs[name] for name in names]], dtype=float),
        base_value_kw=0.0,
        prediction_kw=np.array([sum(mean_abs.values())], dtype=float),
        n_background=100,
        efficiency_error_kw=0.0,
        off_manifold_fraction={},
    )


def test_comparison_ranks_each_column_independently() -> None:
    explanation = _explanation(a=30.0, b=10.0, c=20.0)
    permutation = {"a": 1.0, "b": 5.0, "c": -2.0}

    comparison = explanation_comparison(explanation, permutation)

    assert list(comparison.index) == ["a", "c", "b"]
    assert list(comparison["attribution rank"]) == [1, 2, 3]
    assert list(comparison["permutation rank"]) == [2, 3, 1]


def test_comparison_rejects_a_feature_mismatch() -> None:
    with pytest.raises(ValueError, match="features differ"):
        explanation_comparison(_explanation(a=1.0, b=2.0), {"a": 1.0, "z": 2.0})


def test_rank_agreement_is_one_when_the_two_orders_match() -> None:
    comparison = explanation_comparison(
        _explanation(a=30.0, b=20.0, c=10.0), {"a": 3.0, "b": 2.0, "c": 1.0}
    )

    assert rank_agreement(comparison) == pytest.approx(1.0)


# --------------------------------------------------------------------
# the finding this module exists to make
# --------------------------------------------------------------------


def test_attribution_ranks_a_model_that_learnt_nothing() -> None:
    """THE test. Attribution cannot tell a real correction from noise.

    The target is the residual with its hours shuffled, so no feature
    carries any information about it. The held-out permutation cost sees
    this and reports at most a fraction of a kW for every feature. The
    attribution does not see it: it still assigns kilowatts and still
    produces a top-ranked feature, because it is measuring how the model
    moves its own output, and a model fitted to noise still moves.
    """
    features, residual = _synthetic_residual_year(hours=4000)
    shuffled = set_seed().permutation(residual)
    matrix = features.to_numpy(dtype=float)
    model = default_model_factory()()
    model.fit(matrix, shuffled)

    explanation = shapley_values_kw(
        model, sample_rows(features, 60), sample_rows(features, 60, seed=1), label="shuffled"
    )
    folds = expanding_window_folds(len(features), n_folds=3, spin_up_hours=0, embargo_hours=168)
    permutation = permutation_importance_kw(features, shuffled, folds)

    attributed = explanation.mean_abs_kw
    assert attributed[explanation.ranking[0]] > 1.0
    assert max(permutation.values()) < 1.0
    # And the two disagree about the order, which is the symptom a
    # reviewer can see without knowing the answer.
    assert rank_agreement(explanation_comparison(explanation, permutation)) < 1.0

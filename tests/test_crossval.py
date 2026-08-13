"""Tests for time-series cross-validation (calibration/crossval.py).

Known-answer throughout. The fold layout is exact arithmetic, so it is
asserted as exact arithmetic rather than as properties:

    n_samples = 1000, n_folds = 4  ->  block = 1000 // 5 = 200
    fold 1: train [0, 200)   validate [200, 400)
    fold 2: train [0, 400)   validate [400, 600)
    fold 3: train [0, 600)   validate [600, 800)
    fold 4: train [0, 800)   validate [800, 1000)   <- takes the remainder

The leakage tests use a series with the one property that makes random
splits invalid -- smoothness -- and check the CONSEQUENCE rather than
the mechanism: a model that knows nothing scores well under a random
split and badly under a blocked one, on identical data.
"""

from __future__ import annotations

import numpy as np
import pytest

from cooling_twin.calibration.crossval import (
    DEFAULT_SPIN_UP_HOURS,
    MIN_VALIDATE_HOURS,
    OVERFITTING_GAP_PCT,
    CrossValidationResult,
    TimeFold,
    cross_validate,
    expanding_window_folds,
    neighbour_leak_test,
    neighbour_mean_prediction,
    random_folds,
)

N_SAMPLES = 1000
N_FOLDS = 4
BLOCK = N_SAMPLES // (N_FOLDS + 1)  # 200
SPIN_UP = 24


@pytest.fixture(scope="module")
def smooth_series() -> np.ndarray:
    """A smooth, strongly autocorrelated series -- like a load curve."""
    steps = np.arange(N_SAMPLES)
    return 1000.0 + 300.0 * np.sin(2 * np.pi * steps / 240.0) + 0.5 * steps


# --- fold layout ----------------------------------------------------------


def test_fold_layout_is_exact() -> None:
    """Known answer: five blocks of 200, the first training-only."""
    folds = expanding_window_folds(
        N_SAMPLES, n_folds=N_FOLDS, spin_up_hours=SPIN_UP, min_validate_hours=BLOCK
    )
    assert len(folds) == N_FOLDS
    assert [(f.train_start, f.train_stop) for f in folds] == [
        (0, 200),
        (0, 400),
        (0, 600),
        (0, 800),
    ]
    assert [(f.validate_start, f.validate_stop) for f in folds] == [
        (200, 400),
        (400, 600),
        (600, 800),
        (800, 1000),
    ]


def test_training_never_contains_the_future() -> None:
    """The property the whole module exists for."""
    folds = expanding_window_folds(
        N_SAMPLES, n_folds=N_FOLDS, spin_up_hours=SPIN_UP, min_validate_hours=BLOCK
    )
    for fold in folds:
        assert fold.train_stop <= fold.validate_start


def test_the_last_fold_consumes_the_remainder() -> None:
    """Integer division must not silently drop the end of the year.

    8,760 // 5 = 1,752 and 1,752 * 5 = 8,760 exactly, so the ordinary
    year hides this. A cleaned year does not: 8,782 hours leaves 2 over.
    """
    odd = 8782
    folds = expanding_window_folds(odd, n_folds=N_FOLDS)
    assert folds[-1].validate_stop == odd


def test_spin_up_starts_inside_the_training_region() -> None:
    """Spin-up may use training drivers -- never validation targets."""
    folds = expanding_window_folds(
        N_SAMPLES, n_folds=N_FOLDS, spin_up_hours=SPIN_UP, min_validate_hours=BLOCK
    )
    for fold in folds:
        assert fold.spin_up_start >= fold.train_start
        assert fold.spin_up_start < fold.validate_start
        assert fold.n_spin_up == SPIN_UP
        # And the scored window sits exactly `scored_offset` into the
        # simulated one -- the index that silently mis-scores everything
        # if it is wrong.
        assert fold.scored_offset == SPIN_UP
        simulated = np.arange(fold.spin_up_start, fold.validate_stop)
        assert simulated[fold.scored_offset] == fold.validate_start


def test_embargo_removes_hours_between_train_and_validate() -> None:
    """Known answer: a 50-hour embargo moves every validation start by 50."""
    embargoed = expanding_window_folds(
        N_SAMPLES,
        n_folds=N_FOLDS,
        spin_up_hours=SPIN_UP,
        embargo_hours=50,
        min_validate_hours=BLOCK - 50,
    )
    for fold in embargoed:
        assert fold.validate_start - fold.train_stop == 50


def test_default_spin_up_covers_many_time_constants() -> None:
    """72 h against L4.3's measured ~4.9 h slow mode.

    Pinned as a test because the number is a physics claim, not a
    preference: if someone lowers it, this fails and they have to say
    which time constant they are relying on.
    """
    slow_mode_hours = 4.9
    assert DEFAULT_SPIN_UP_HOURS / slow_mode_hours > 10.0


@pytest.mark.parametrize("n_folds", [0, 1, -3])
def test_fewer_than_two_folds_rejected(n_folds: int) -> None:
    with pytest.raises(ValueError, match="n_folds must be >= 2"):
        expanding_window_folds(N_SAMPLES, n_folds=n_folds)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"spin_up_hours": -1}, "spin_up_hours must be >= 0"),
        ({"embargo_hours": -1}, "embargo_hours must be >= 0"),
        ({"min_validate_hours": -1}, "min_validate_hours must be >= 0"),
    ],
)
def test_negative_hour_counts_rejected(kwargs: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        expanding_window_folds(N_SAMPLES, n_folds=N_FOLDS, **kwargs)


def test_too_short_a_validation_window_is_refused_not_shrunk() -> None:
    """A caller that asked for 8 folds and got 2 would report 8."""
    with pytest.raises(ValueError, match="below the .* minimum"):
        expanding_window_folds(2000, n_folds=8, min_validate_hours=MIN_VALIDATE_HOURS)


def test_spin_up_longer_than_the_first_block_is_refused() -> None:
    """The earliest fold must be able to spin up from its own training data."""
    with pytest.raises(ValueError, match="exceeds the first training block"):
        expanding_window_folds(
            N_SAMPLES, n_folds=N_FOLDS, spin_up_hours=BLOCK + 1, min_validate_hours=BLOCK
        )


# --- the leak -------------------------------------------------------------


def test_a_random_split_lets_a_knowledge_free_model_win(
    smooth_series: np.ndarray,
) -> None:
    """The curriculum's claim, measured rather than asserted.

    Same series, same control model, two splits.

    What is asserted is the RATIO, not an absolute threshold on the
    blocked score: how badly the control does under a blocked split
    depends on the amplitude of whatever series it is given, so a
    "> 30%" assertion here would be a number chosen to pass rather than
    a number derived. The leak is that the SAME model, on the SAME data,
    scores an order of magnitude better purely because of how the split
    was drawn -- and that under the random split it clears G14's hourly
    limit while knowing nothing about the building.
    """
    scattered = random_folds(N_SAMPLES, n_folds=N_FOLDS)
    blocked = [
        np.arange(fold.validate_start, fold.validate_stop)
        for fold in expanding_window_folds(
            N_SAMPLES, n_folds=N_FOLDS, spin_up_hours=SPIN_UP, min_validate_hours=BLOCK
        )
    ]

    leaky = neighbour_leak_test(smooth_series, scattered)
    honest = neighbour_leak_test(smooth_series, blocked)

    assert leaky < 30.0
    assert honest > 10.0 * leaky


def test_the_neighbour_model_may_not_use_held_out_values() -> None:
    """The bug that makes the diagnostic useless, pinned.

    Reading `values[position - 1]` unconditionally scores a blocked
    split as well as a random one, because both then read held-out
    truth. Known answer: on a straight line, a point whose neighbours
    are BOTH held out must be predicted from the available points
    outside the block, not from its immediate neighbours.
    """
    line = np.arange(20.0)
    held_out = np.arange(5, 15)  # a contiguous block
    predicted = neighbour_mean_prediction(line, held_out)

    # Nearest available points are 4 and 15 for every held-out position,
    # so every prediction is their mean -- constant, and wrong in the
    # middle. An implementation reading immediate neighbours would
    # return the exact truth here instead.
    assert predicted == pytest.approx(np.full(held_out.size, (4.0 + 15.0) / 2.0))
    assert not np.allclose(predicted, line[held_out])


def test_the_neighbour_model_is_exact_on_a_line_when_isolated() -> None:
    """Control for the control: with both neighbours available it is exact."""
    line = np.arange(20.0)
    isolated = np.array([3, 9, 16])
    assert neighbour_mean_prediction(line, isolated) == pytest.approx(line[isolated])


def test_edge_positions_fall_back_to_one_side() -> None:
    """No averaging a value with itself at the ends of the series."""
    line = np.arange(10.0)
    assert neighbour_mean_prediction(line, [0]) == pytest.approx([1.0])
    assert neighbour_mean_prediction(line, [9]) == pytest.approx([8.0])


def test_holding_out_everything_is_refused() -> None:
    with pytest.raises(ValueError, match="every point is held out"):
        neighbour_mean_prediction(np.arange(5.0), np.arange(5))


def test_out_of_range_positions_rejected() -> None:
    with pytest.raises(ValueError, match="must lie inside values"):
        neighbour_mean_prediction(np.arange(5.0), [7])


def test_a_two_point_minimum_is_enforced() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        neighbour_mean_prediction([1.0], [0])


def test_leak_test_needs_something_held_out(smooth_series: np.ndarray) -> None:
    with pytest.raises(ValueError, match="at least one fold"):
        neighbour_leak_test(smooth_series, [])
    with pytest.raises(ValueError, match="all empty"):
        neighbour_leak_test(smooth_series, [np.array([], dtype=int)])


@pytest.mark.parametrize("n_folds", [1, N_SAMPLES + 1])
def test_random_folds_validates_its_count(n_folds: int) -> None:
    with pytest.raises(ValueError, match="n_folds must be in"):
        random_folds(N_SAMPLES, n_folds=n_folds)


def test_random_folds_partition_the_series() -> None:
    """Every position held out exactly once, whatever the seed."""
    parts = random_folds(N_SAMPLES, n_folds=N_FOLDS, seed=11)
    pooled = np.concatenate(parts)
    assert np.array_equal(np.sort(pooled), np.arange(N_SAMPLES))


# --- scoring --------------------------------------------------------------


def _constant_model(series: np.ndarray) -> tuple:
    """Fit/predict pair for a one-parameter model: the training mean."""

    def fit(fold: TimeFold) -> dict[str, float]:
        return {"level": float(np.mean(series[fold.train_slice]))}

    def predict(parameters: dict[str, float], window: slice) -> np.ndarray:
        return np.full(len(series[window]), parameters["level"])

    return fit, predict


def test_cross_validate_scores_the_right_hours(smooth_series: np.ndarray) -> None:
    """Spin-up predictions must be dropped, not scored.

    Known answer: a constant model's CV(RMSE) on the validation window
    is computable directly, so any off-by-one in the spin-up handling
    changes the number.
    """
    folds = expanding_window_folds(
        N_SAMPLES, n_folds=N_FOLDS, spin_up_hours=SPIN_UP, min_validate_hours=BLOCK
    )
    fit, predict = _constant_model(smooth_series)
    result = cross_validate(fit, predict, smooth_series, folds, n_params=1)

    assert len(result.scores) == N_FOLDS
    for score in result.scores:
        window = smooth_series[score.fold.validate_slice]
        level = score.parameters["level"]
        expected = (
            100.0
            * np.sqrt(np.sum((window - level) ** 2) / (window.size - 1))
            / window.mean()
        )
        assert score.validate_cvrmse_pct == pytest.approx(expected)


def test_a_length_mismatch_is_caught_not_scored(smooth_series: np.ndarray) -> None:
    """The worst failure mode in the file: silently scoring wrong hours."""
    folds = expanding_window_folds(
        N_SAMPLES, n_folds=N_FOLDS, spin_up_hours=SPIN_UP, min_validate_hours=BLOCK
    )
    fit, predict = _constant_model(smooth_series)

    def truncated(parameters: dict[str, float], window: slice) -> np.ndarray:
        return predict(parameters, window)[:-1]

    with pytest.raises(ValueError, match="predict\\(\\) returned"):
        cross_validate(fit, truncated, smooth_series, folds, n_params=1)


def test_cross_validate_needs_folds(smooth_series: np.ndarray) -> None:
    fit, predict = _constant_model(smooth_series)
    with pytest.raises(ValueError, match="folds must not be empty"):
        cross_validate(fit, predict, smooth_series, [], n_params=1)


def _score(train: float, validate: float, number: int = 1):
    """A FoldScore with the two CV(RMSE) values under test."""
    from cooling_twin.calibration.crossval import FoldScore

    fold = TimeFold(number, 0, 100, 90, 100, 200)
    return FoldScore(
        fold=fold,
        train_cvrmse_pct=train,
        validate_cvrmse_pct=validate,
        train_nmbe_pct=0.0,
        validate_nmbe_pct=0.0,
        parameters={"level": float(number)},
    )


def test_the_gap_is_validation_minus_training() -> None:
    """Sign convention: positive means it did worse on unseen data."""
    assert _score(10.0, 18.0).gap_pct == pytest.approx(8.0)
    assert _score(18.0, 10.0).gap_pct == pytest.approx(-8.0)


def test_overfitting_verdict_uses_the_declared_threshold() -> None:
    """Exactly at the threshold does not trip it; above does."""
    at_limit = CrossValidationResult(
        scores=(_score(10.0, 10.0 + OVERFITTING_GAP_PCT),), parameter_names=("level",)
    )
    over = CrossValidationResult(
        scores=(_score(10.0, 10.1 + OVERFITTING_GAP_PCT),), parameter_names=("level",)
    )
    assert not at_limit.overfitting_suspected
    assert over.overfitting_suspected


def test_worst_fold_is_reported_beside_the_mean() -> None:
    """A mean hides the fold that failed, and that fold is the finding."""
    result = CrossValidationResult(
        scores=(_score(10.0, 12.0, 1), _score(10.0, 40.0, 2)),
        parameter_names=("level",),
    )
    assert result.mean_validate_cvrmse_pct == pytest.approx(26.0)
    assert result.worst_validate_cvrmse_pct == pytest.approx(40.0)


def test_parameter_stability_brackets_every_fold() -> None:
    """L6.8's identifiability question, asked with independent fits."""
    result = CrossValidationResult(
        scores=(_score(10.0, 12.0, 1), _score(10.0, 14.0, 7)),
        parameter_names=("level",),
    )
    assert result.parameter_stability()["level"] == (1.0, 7.0)


def test_a_fold_inside_cvrmse_but_outside_nmbe_fails_g14() -> None:
    """L6.2's NMBE trap, in cross-validation costume.

    Known answer against G14's hourly limits (+-10% NMBE, 30% CV(RMSE)):
    a fold at 15.7% CV(RMSE) is comfortably inside the scatter limit and
    a bias of +11.9% is outside the bias limit, so the fold FAILS. A
    summary reporting CV(RMSE) alone would call it a pass -- which is
    what the real Claude run did on fold 2 before this was added.
    """
    from cooling_twin.calibration.crossval import FoldScore

    fold = TimeFold(2, 0, 100, 90, 100, 200)
    biased = FoldScore(
        fold=fold,
        train_cvrmse_pct=9.22,
        validate_cvrmse_pct=15.68,
        train_nmbe_pct=0.0,
        validate_nmbe_pct=11.93,
        parameters={"level": 1.0},
    )
    unbiased = FoldScore(
        fold=fold,
        train_cvrmse_pct=9.22,
        validate_cvrmse_pct=15.68,
        train_nmbe_pct=0.0,
        validate_nmbe_pct=-9.99,
        parameters={"level": 1.0},
    )
    assert not biased.passes_g14()
    assert unbiased.passes_g14()

    result = CrossValidationResult(scores=(biased, unbiased), parameter_names=("level",))
    assert result.folds_failing_g14 == (2,)
    assert "G14 FAILED on held-out fold(s): 2" in result.summary()


def test_a_small_gap_does_not_imply_an_acceptable_fold() -> None:
    """The two verdicts are independent and both get reported.

    A model can generalise perfectly -- zero gap -- and fail the
    standard on both windows. Conflating "generalises" with "good" is
    the reading this guards against.
    """
    from cooling_twin.calibration.crossval import FoldScore

    bad_but_consistent = FoldScore(
        fold=TimeFold(1, 0, 100, 90, 100, 200),
        train_cvrmse_pct=45.0,
        validate_cvrmse_pct=45.0,
        train_nmbe_pct=0.0,
        validate_nmbe_pct=0.0,
        parameters={"level": 1.0},
    )
    result = CrossValidationResult(
        scores=(bad_but_consistent,), parameter_names=("level",)
    )
    assert result.mean_gap_pct == pytest.approx(0.0)
    assert not result.overfitting_suspected
    assert result.folds_failing_g14 == (1,)


def test_summary_names_the_verdict_and_the_threshold() -> None:
    """The report line has to carry the threshold it was judged against."""
    text = CrossValidationResult(
        scores=(_score(10.0, 30.0),), parameter_names=("level",)
    ).summary()
    assert "OVERFITTING SUSPECTED" in text
    assert "5.0 pp" in text

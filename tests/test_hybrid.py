"""Tests for the hybrid physics + ML residual model (L7.3).

Two kinds of test, deliberately separated.

Most of this file uses `_MeanResidualModel`, a stub learner that
predicts the mean of whatever it was fitted on. Its answer is derivable
on paper, so a failure localises to the plumbing under test -- the fold
partition, the mask, the clip, the arithmetic -- and never to a tree
ensemble's internals. This is the same pattern L6.5 used for Morris: fix
the part whose behaviour is known, and test the machinery around it.

The rest fit the real gradient booster on data whose answer was INJECTED
before the fit. Those tests check the claim the module exists to make --
that the reported ML share is what the model actually learnt about the
building, not what it memorised about the year -- and the sharpest of
them is `test_white_noise_residual_earns_no_out_of_fold_credit`, which
proves the out-of-fold discipline has teeth by handing it a residual
with nothing in it at all.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from cooling_twin import set_seed
from cooling_twin.analysis.hybrid import (
    DEFAULT_N_FOLDS,
    FEATURE_NAMES,
    MIN_TRAIN_HOURS,
    WEATHER_WINDOW_HOURS,
    HybridResult,
    build_features,
    fit_hybrid,
    out_of_fold_correction,
    permutation_importance_kw,
    variance_decomposition,
)
from cooling_twin.calibration.crossval import TimeFold, expanding_window_folds


class _MeanResidualModel:
    """A learner whose prediction is known on paper: the training mean."""

    def __init__(self) -> None:
        self.value = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> _MeanResidualModel:
        self.value = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


def _mean_factory() -> _MeanResidualModel:
    return _MeanResidualModel()


class _ConstantModel:
    """A learner that ignores its training data and always predicts +50 kW.

    Used where a test needs the correction to be a KNOWN number rather
    than a fitted one, so the resulting sums of squares can be written
    out on paper.
    """

    CORRECTION_KW = 50.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> _ConstantModel:
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.CORRECTION_KW, dtype=float)


def _constant_factory() -> _ConstantModel:
    return _ConstantModel()


def _synthetic_year(
    hours: int = 8760, noise_kw: float = 90.0
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A year whose physics model is missing two known switched terms.

    Returns:
        `(index, measured, physics, t_outdoor, humidity_ratio)`.
    """
    rng = set_seed()
    index = pd.date_range("2016-01-01", periods=hours, freq="h")
    steps = np.arange(hours, dtype=float)
    t_outdoor = (
        20.0
        + 12.0 * np.sin(2.0 * np.pi * (steps - 2000.0) / 8760.0)
        + 5.0 * np.sin(2.0 * np.pi * steps / 24.0)
    )
    humidity = 0.006 + 0.004 * np.sin(2.0 * np.pi * (steps - 2000.0) / 8760.0)
    physics = 400.0 + 30.0 * t_outdoor
    measured = (
        physics
        + 40.0 * np.maximum(t_outdoor - 28.0, 0.0)
        + 25.0 * np.maximum(humidity * 1000.0 - 8.0, 0.0)
        + rng.normal(0.0, noise_kw, size=hours)
    )
    return index, measured, physics, t_outdoor, humidity


# --------------------------------------------------------------------
# variance_decomposition -- known answers, hand-derived
# --------------------------------------------------------------------


def test_variance_decomposition_known_answer() -> None:
    """Every share checked against arithmetic done on paper.

    measured = [0, 2, 4, 6], mean 3
        ss_total   = 9 + 1 + 1 + 9 = 20
    physics  = [1, 2, 3, 4]      errors -1, 0, 1, 2
        ss_physics = 1 + 0 + 1 + 4 =  6   -> physics 100*(1 - 6/20) = 70%
    hybrid   = [0, 2, 5, 6]      errors  0, 0, -1, 0
        ss_hybrid  = 0 + 0 + 1 + 0 =  1   -> ML      100*(6 - 1)/20 = 25%
                                             unexplained 100*1/20   =  5%
    """
    result = variance_decomposition(
        [0.0, 2.0, 4.0, 6.0], [1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 5.0, 6.0]
    )

    assert result.ss_total == pytest.approx(20.0)
    assert result.ss_physics == pytest.approx(6.0)
    assert result.ss_hybrid == pytest.approx(1.0)
    assert result.physics_pct == pytest.approx(70.0)
    assert result.ml_pct == pytest.approx(25.0)
    assert result.unexplained_pct == pytest.approx(5.0)
    assert result.explained_pct == pytest.approx(95.0)
    assert result.n_hours == 4


def test_shares_sum_to_one_hundred_on_random_data() -> None:
    """The three shares are a partition, not three separate estimates."""
    rng = set_seed()
    for _ in range(50):
        measured = rng.normal(1000.0, 200.0, size=200)
        physics = measured + rng.normal(0.0, 150.0, size=200)
        hybrid = measured + rng.normal(0.0, 100.0, size=200)
        result = variance_decomposition(measured, physics, hybrid)
        total = result.physics_pct + result.ml_pct + result.unexplained_pct
        assert total == pytest.approx(100.0)


def test_physics_share_is_negative_when_physics_loses_to_the_mean(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """L6.4 measured exactly this on the uncalibrated model -- it must not be hidden."""
    measured = np.array([0.0, 2.0, 4.0, 6.0])
    physics = np.full(4, 50.0)
    with caplog.at_level(logging.WARNING, logger="cooling_twin.analysis.hybrid"):
        result = variance_decomposition(measured, physics, physics)
    assert result.physics_pct < 0.0
    assert "NEGATIVE" in caplog.text


def test_ml_dominance_is_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """A correction larger than the physics changes what the artifact IS."""
    measured = np.array([0.0, 2.0, 4.0, 6.0])
    physics = np.array([3.0, 3.0, 3.0, 3.0])  # the mean: explains nothing
    hybrid = np.array([0.0, 2.0, 4.0, 6.0])  # perfect
    with caplog.at_level(logging.WARNING, logger="cooling_twin.analysis.hybrid"):
        result = variance_decomposition(measured, physics, hybrid)
    assert result.ml_dominates
    assert "no longer a physical model" in caplog.text


@pytest.mark.parametrize(
    ("measured", "physics", "hybrid", "message"),
    [
        ([1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0, 3.0], "points"),
        ([1.0, 2.0, 3.0], [1.0, np.nan, 3.0], [1.0, 2.0, 3.0], "non-finite"),
        ([1.0], [1.0], [1.0], "at least two points"),
        ([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], "zero variance"),
    ],
)
def test_variance_decomposition_rejects_bad_input(
    measured: list[float], physics: list[float], hybrid: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        variance_decomposition(measured, physics, hybrid)


# --------------------------------------------------------------------
# build_features
# --------------------------------------------------------------------


def test_feature_frame_has_exactly_the_declared_columns_in_order() -> None:
    """A reordered matrix silently retrains on different features."""
    index = pd.date_range("2016-01-01", periods=500, freq="h")
    features = build_features(index, np.full(500, 20.0), np.full(500, 0.008))
    assert tuple(features.columns) == FEATURE_NAMES
    assert features.index.equals(index)


def test_hour_harmonics_make_midnight_adjacent_to_2300() -> None:
    """The reason hour of day is not an integer column.

    On the unit circle hour 23 and hour 0 are one step apart, while hour
    0 and hour 12 are as far apart as it is possible to be. An integer
    column reverses that.
    """
    index = pd.date_range("2016-01-01", periods=24, freq="h")
    features = build_features(index, np.full(24, 20.0), np.full(24, 0.008))
    circle = features[["hour_sin", "hour_cos"]].to_numpy()

    adjacent = np.linalg.norm(circle[23] - circle[0])
    opposite = np.linalg.norm(circle[12] - circle[0])
    assert adjacent < opposite
    assert adjacent == pytest.approx(2.0 * np.sin(np.pi / 24.0))


def test_lagged_weather_is_computed_on_the_clock_not_on_rows() -> None:
    """The M3-gap trap: a row-based window spans more hours than it claims.

    200 hours with a 21-hour hole cut out of the middle. At the first row
    AFTER the hole, only three of the previous 24 CLOCK hours survive in
    the data, so the feature must be NaN. A `rolling(24)` applied to the
    gappy frame sees 24 rows there and returns a confident number built
    from three days.
    """
    grid = pd.date_range("2016-01-01", periods=200, freq="h")
    index = grid.delete(range(100, 121))
    temperature = np.arange(index.size, dtype=float)

    features = build_features(index, temperature, np.full(index.size, 0.008))
    lagged = features["outdoor_dry_bulb_24h_mean_c"].to_numpy()

    first_after_gap = int(np.flatnonzero(index == grid[121])[0])
    assert np.isnan(lagged[first_after_gap])
    # Well away from the hole the feature is the plain 24-hour mean.
    assert lagged[99] == pytest.approx(temperature[99 - WEATHER_WINDOW_HOURS + 1 : 100].mean())


def test_lagged_weather_leads_with_nan_rather_than_a_short_mean() -> None:
    """The first hours of a year have no previous day to average."""
    index = pd.date_range("2016-01-01", periods=48, freq="h")
    features = build_features(index, np.arange(48, dtype=float), np.full(48, 0.008))
    lagged = features["outdoor_dry_bulb_24h_mean_c"].to_numpy()
    assert np.isnan(lagged[:11]).all()
    assert np.isfinite(lagged[11:]).all()


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (pd.RangeIndex(10), "DatetimeIndex"),
        (pd.DatetimeIndex([]), "empty"),
        (pd.DatetimeIndex(["2016-01-02", "2016-01-01"]), "sorted"),
        (pd.DatetimeIndex(["2016-01-01", "2016-01-01"]), "duplicate"),
    ],
)
def test_build_features_rejects_bad_index(index: pd.Index, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_features(index, np.zeros(len(index)), np.zeros(len(index)))  # type: ignore[arg-type]


def test_build_features_rejects_non_finite_weather() -> None:
    index = pd.date_range("2016-01-01", periods=48, freq="h")
    temperature = np.full(48, 20.0)
    temperature[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_features(index, temperature, np.full(48, 0.008))


def test_build_features_rejects_length_mismatch() -> None:
    index = pd.date_range("2016-01-01", periods=48, freq="h")
    with pytest.raises(ValueError, match="shape"):
        build_features(index, np.full(47, 20.0), np.full(48, 0.008))


# --------------------------------------------------------------------
# out_of_fold_correction -- the fold partition and the mask
# --------------------------------------------------------------------


def test_out_of_fold_correction_matches_hand_computed_fold_means() -> None:
    """With a mean-predicting learner the correction is arithmetic.

    Each fold's correction must equal the mean residual of that fold's
    TRAINING block -- not of its scored block, which would be the leak.
    """
    n = 4000
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    features = build_features(index, np.full(n, 20.0), np.full(n, 0.008))
    residual = np.arange(n, dtype=float)
    folds = expanding_window_folds(n, n_folds=2, spin_up_hours=0, embargo_hours=0)

    correction, scored = out_of_fold_correction(
        features, residual, folds, model_factory=_mean_factory
    )

    for fold in folds:
        expected = residual[fold.train_slice].mean()
        assert np.allclose(correction[fold.validate_slice], expected)
        assert scored[fold.validate_slice].all()
    # Fold 1's training block is never scored by anything.
    assert not scored[folds[0].train_slice].any()


def test_embargoed_hours_are_left_unscored() -> None:
    """The gap between training and scoring belongs to neither."""
    n = 6000
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    features = build_features(index, np.full(n, 20.0), np.full(n, 0.008))
    folds = expanding_window_folds(n, n_folds=3, spin_up_hours=0, embargo_hours=168)

    _correction, scored = out_of_fold_correction(
        features, np.zeros(n), folds, model_factory=_mean_factory
    )

    embargo = slice(folds[0].validate_stop, folds[1].validate_start)
    assert embargo.stop > embargo.start
    assert not scored[embargo].any()
    assert scored.sum() == sum(fold.n_validate for fold in folds)


def test_out_of_fold_correction_refuses_a_starved_training_block() -> None:
    """A fold trained on days cannot support a boosted ensemble.

    The fold is hand-built rather than taken from
    `expanding_window_folds`, because that function's own
    `min_validate_hours` floor of 336 h makes fold 1's training block at
    least 336 h -- it cannot produce a starved fold. This guard exists
    for the caller who passes folds of their own, which is the only way
    the situation arises, and testing it any other way would be testing
    that crossval's floor still holds rather than that this one does.
    """
    n = 4000
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    features = build_features(index, np.full(n, 20.0), np.full(n, 0.008))
    starved = TimeFold(
        number=1,
        train_start=0,
        train_stop=MIN_TRAIN_HOURS - 1,
        spin_up_start=MIN_TRAIN_HOURS - 1,
        validate_start=MIN_TRAIN_HOURS - 1,
        validate_stop=n,
    )
    with pytest.raises(ValueError, match="below the"):
        out_of_fold_correction(
            features, np.zeros(n), (starved,), model_factory=_mean_factory
        )


def test_out_of_fold_correction_rejects_bad_input() -> None:
    n = 4000
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    features = build_features(index, np.full(n, 20.0), np.full(n, 0.008))
    folds = expanding_window_folds(n, n_folds=2, spin_up_hours=0, embargo_hours=0)

    with pytest.raises(ValueError, match="points"):
        out_of_fold_correction(features, np.zeros(n - 1), folds, model_factory=_mean_factory)
    with pytest.raises(ValueError, match="non-finite"):
        residual = np.zeros(n)
        residual[3] = np.inf
        out_of_fold_correction(features, residual, folds, model_factory=_mean_factory)
    with pytest.raises(ValueError, match="no folds"):
        out_of_fold_correction(features, np.zeros(n), (), model_factory=_mean_factory)


# --------------------------------------------------------------------
# fit_hybrid -- composition, the clip, and the reported numbers
# --------------------------------------------------------------------


def _hybrid_with_stub(measured: np.ndarray, physics: np.ndarray) -> HybridResult:
    n = measured.size
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    return fit_hybrid(
        index,
        measured,
        physics,
        t_outdoor_c=np.full(n, 20.0),
        humidity_ratio_kg_per_kg=np.full(n, 0.008),
        n_physics_params=5,
        n_folds=2,
        embargo_hours=0,
        model_factory=_mean_factory,
        label="stub",
    )


def test_hybrid_is_physics_plus_correction_clipped_at_zero() -> None:
    """A cooling load cannot be negative, correction or no correction."""
    rng = set_seed()
    n = 6000
    physics = rng.uniform(0.0, 50.0, size=n)
    measured = physics - 200.0 + rng.normal(0.0, 10.0, size=n)  # forces a negative hybrid

    result = _hybrid_with_stub(measured, physics)

    assert (result.hybrid_kw >= 0.0).all()
    expected = np.maximum(physics + result.correction_kw, 0.0)
    assert np.allclose(result.hybrid_kw, expected)
    assert result.clipped_fraction > 0.0


def test_no_clipping_is_reported_when_none_happened() -> None:
    rng = set_seed()
    n = 6000
    physics = rng.uniform(2000.0, 3000.0, size=n)
    measured = physics + rng.normal(0.0, 50.0, size=n)
    assert _hybrid_with_stub(measured, physics).clipped_fraction == 0.0


def test_a_pooled_gain_hiding_a_harmed_fold_is_flagged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A POSITIVE pooled share can hide a block the correction ruined.

    Hand-derived, with a learner that always predicts +50 kW so no fit
    is involved. Two folds of 2,000 scored hours each:

        fold 1   measured - physics = +80  ->  physics err 80, hybrid 30
                 ss gain per hour = 80^2 - 30^2 = 6400 - 900 = 5500
        fold 2   measured - physics =   0  ->  physics err  0, hybrid 50
                 ss loss per hour  =  0   - 50^2            = -2500

    Pooled, the +5,500 block outweighs the -2,500 one and the ML share
    comes out POSITIVE. Fold 2 is still a block the correction made
    worse, and nothing in the pooled number says so.
    """
    n = 6000
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    steps = np.arange(n, dtype=float)
    # Non-constant, so each block has variance for the share's denominator.
    physics = 2000.0 + 300.0 * np.sin(2.0 * np.pi * steps / 1000.0)
    measured = physics.copy()
    folds = expanding_window_folds(n, n_folds=2, spin_up_hours=0, embargo_hours=0)
    measured[folds[0].validate_slice] += 80.0

    with caplog.at_level(logging.WARNING, logger="cooling_twin.analysis.hybrid"):
        result = fit_hybrid(
            index,
            measured,
            physics,
            t_outdoor_c=np.full(n, 20.0),
            humidity_ratio_kg_per_kg=np.full(n, 0.008),
            n_physics_params=5,
            n_folds=2,
            embargo_hours=0,
            model_factory=_constant_factory,
            label="disagreeing folds",
        )

    assert len(result.fold_ml_pct) == 2
    assert result.fold_ml_pct[0] > 0.0
    assert result.fold_ml_pct[1] < 0.0
    assert result.n_folds_harmed == 1
    assert result.decomposition.ml_pct > 0.0  # the pooled number hides it
    assert "WORSE" in caplog.text


def test_no_flag_when_every_fold_is_helped() -> None:
    """The flag must stay silent on a correction that works everywhere."""
    n = 6000
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    steps = np.arange(n, dtype=float)
    physics = 2000.0 + 300.0 * np.sin(2.0 * np.pi * steps / 1000.0)
    measured = physics + 80.0  # the +50 kW correction closes most of it

    result = fit_hybrid(
        index,
        measured,
        physics,
        t_outdoor_c=np.full(n, 20.0),
        humidity_ratio_kg_per_kg=np.full(n, 0.008),
        n_physics_params=5,
        n_folds=2,
        embargo_hours=0,
        model_factory=_constant_factory,
        label="uniform",
    )
    assert result.n_folds_harmed == 0
    assert all(share > 0.0 for share in result.fold_ml_pct)


def test_a_seasonal_correction_harms_some_folds_by_construction() -> None:
    """Expanding-window folds SYSTEMATICALLY understate a seasonal term.

    This is not a defect being tolerated, it is the estimator's known
    bias, and it is why `fold_ml_pct` is reported next to the pooled
    share rather than instead of it. The injected term only switches on
    above 28 degC. A fold trained on winter cannot learn it; a fold
    scored on winter has nothing for it to correct and pays for the
    correction's noise. Both show up as harmed folds even though the
    pooled share is solidly positive and the underlying term is real.

    The same pattern appears on the real buildings -- see the per-fold
    shares in reports/calibration_runs/hybrid_2016.json.
    """
    index, measured, physics, t_outdoor, humidity = _synthetic_year()
    result = fit_hybrid(
        index,
        measured,
        physics,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=humidity,
        n_physics_params=5,
        label="seasonal",
    )
    assert len(result.fold_ml_pct) == DEFAULT_N_FOLDS
    assert result.decomposition.ml_pct > 0.0
    assert result.n_folds_harmed > 0


def test_reported_numbers_cover_only_the_scored_hours() -> None:
    """Physics and ML must be scored on the same sample, or neither means anything."""
    rng = set_seed()
    n = 6000
    physics = rng.uniform(2000.0, 3000.0, size=n)
    measured = physics + rng.normal(0.0, 200.0, size=n)

    result = _hybrid_with_stub(measured, physics)

    assert result.n_hours_total == n
    assert result.n_hours_scored == int(result.scored_mask.sum())
    assert result.n_hours_scored < n
    assert result.decomposition.n_hours == result.n_hours_scored
    assert 0.0 < result.scored_fraction < 1.0


# --------------------------------------------------------------------
# The real learner, on data whose answer was injected first
# --------------------------------------------------------------------


def test_white_noise_residual_earns_no_out_of_fold_credit() -> None:
    """The teeth of the whole module.

    The residual here is pure noise -- there is nothing in it to learn.
    A booster fitted and scored on the same hours will still report a
    healthy ML share, because it can memorise. Out of fold it must earn
    approximately nothing, and is allowed to earn LESS than nothing.

    If this test ever passes trivially because both numbers are near
    zero, check that the in-sample assertion below still fires: it is
    what proves the model was capable of memorising and the fold
    partition is what stopped it.
    """
    rng = set_seed()
    n = 8760
    index = pd.date_range("2016-01-01", periods=n, freq="h")
    steps = np.arange(n, dtype=float)
    t_outdoor = 20.0 + 12.0 * np.sin(2.0 * np.pi * steps / 8760.0)
    physics = 400.0 + 30.0 * t_outdoor
    measured = physics + rng.normal(0.0, 100.0, size=n)

    result = fit_hybrid(
        index,
        measured,
        physics,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=np.full(n, 0.008),
        n_physics_params=5,
        label="white noise",
    )

    assert result.decomposition.ml_pct < 0.5
    assert result.in_sample_decomposition.ml_pct > result.decomposition.ml_pct
    assert result.memorisation_gap_pct > 0.0


def test_injected_switched_terms_are_recovered_out_of_fold() -> None:
    """The complement: a residual that IS learnable must be learnt.

    The synthetic year carries a hockey stick above 28 degC and a latent
    term above 8 g/kg, worth ~9% of total variance between them, on top
    of ~6% irreducible noise. The hybrid should recover most of the 9
    and cannot touch the 6.
    """
    index, measured, physics, t_outdoor, humidity = _synthetic_year()

    result = fit_hybrid(
        index,
        measured,
        physics,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=humidity,
        n_physics_params=5,
        label="injected",
    )

    assert result.decomposition.ml_pct > 5.0
    assert result.decomposition.ml_pct < 12.0  # cannot exceed what was injected
    assert result.hybrid_cvrmse_pct < result.physics_cvrmse_pct
    assert result.cvrmse_improvement_pct > 0.0
    assert not result.decomposition.ml_dominates


def test_the_correction_removes_structure_the_physics_left_behind() -> None:
    """L7.2's test, re-run on the corrected residual -- the point of L7.3.

    The daily variance share is the decisive L7.2 number: white noise
    would leave 1/24. If the ML layer took real structure out, the share
    falls. If it fell to 1/24 the model would be finished; it does not,
    and that is M8's problem, not this one's.
    """
    index, measured, physics, t_outdoor, humidity = _synthetic_year()

    result = fit_hybrid(
        index,
        measured,
        physics,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=humidity,
        n_physics_params=5,
        label="structure",
    )

    before = result.diagnostics_before.daily_variance_share
    after = result.diagnostics_after.daily_variance_share
    assert after < before
    assert result.diagnostics_before.survives_daily_averaging


def test_fit_hybrid_is_deterministic_under_the_project_seed() -> None:
    """Two identical runs must produce identical numbers, or nothing is reproducible."""
    index, measured, physics, t_outdoor, humidity = _synthetic_year(hours=4000)
    kwargs = {
        "t_outdoor_c": t_outdoor,
        "humidity_ratio_kg_per_kg": humidity,
        "n_physics_params": 5,
        "n_folds": 3,
    }
    first = fit_hybrid(index, measured, physics, **kwargs)  # type: ignore[arg-type]
    second = fit_hybrid(index, measured, physics, **kwargs)  # type: ignore[arg-type]

    assert first.decomposition.ml_pct == second.decomposition.ml_pct
    assert np.array_equal(first.correction_kw, second.correction_kw)


def test_fit_hybrid_rejects_mismatched_series() -> None:
    index = pd.date_range("2016-01-01", periods=4000, freq="h")
    with pytest.raises(ValueError, match="points"):
        fit_hybrid(
            index,
            np.zeros(4000),
            np.zeros(3999),
            t_outdoor_c=np.full(4000, 20.0),
            humidity_ratio_kg_per_kg=np.full(4000, 0.008),
            n_physics_params=5,
        )


# --------------------------------------------------------------------
# permutation_importance_kw
# --------------------------------------------------------------------


def test_permutation_importance_ranks_the_driver_that_was_injected() -> None:
    """The residual was built from temperature; temperature must lead."""
    index, measured, physics, t_outdoor, humidity = _synthetic_year(hours=6000)
    features = build_features(index, t_outdoor, humidity)
    folds = expanding_window_folds(6000, n_folds=3, spin_up_hours=0, embargo_hours=168)

    importance = permutation_importance_kw(
        features, measured - physics, folds, n_repeats=3
    )

    assert next(iter(importance)) == "outdoor_dry_bulb_c"
    assert importance["outdoor_dry_bulb_c"] > 0.0
    assert set(importance) == set(FEATURE_NAMES)


def test_permutation_importance_rejects_bad_input() -> None:
    index = pd.date_range("2016-01-01", periods=4000, freq="h")
    features = build_features(index, np.full(4000, 20.0), np.full(4000, 0.008))
    folds = expanding_window_folds(4000, n_folds=2, spin_up_hours=0, embargo_hours=0)

    with pytest.raises(ValueError, match="points"):
        permutation_importance_kw(features, np.zeros(3999), folds)
    with pytest.raises(ValueError, match="no folds"):
        permutation_importance_kw(features, np.zeros(4000), ())

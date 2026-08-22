"""Unit tests for conformal prediction (src/cooling_twin/twin/uncertainty.py).

Test patterns 1 and 2 of 4 (05_ENGINEERING_STANDARDS.md SS3): known-answer
tests where the quantile is small enough to compute by hand, property
tests for the coverage guarantee itself, and one input-validation test
per raise branch.

The most important test in this file is
`test_finite_sample_correction_is_not_cosmetic`: it fails if the
`(n + 1)` correction is dropped, which is the single change most likely
to be made by someone "simplifying" this module, and its effect --
under-coverage on small calibration sets -- is invisible without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cooling_twin import SEED
from cooling_twin.twin.uncertainty import (
    DEFAULT_ALPHA,
    ConformalInterval,
    block_bootstrap_ci,
    conformal_interval,
    conformal_quantile,
    coverage_by_group,
    interleaved_block_split,
    mondrian_interval,
    mondrian_quantiles,
    normalising_scale,
    time_ordered_split,
    validate_coverage,
)

# --- conformal_quantile: known answers ------------------------------------


def test_conformal_quantile_known_answer() -> None:
    """Residuals 1..10, alpha=0.1: rank = ceil(11 * 0.9) = 10, so the
    quantile is the 10th smallest score = 10.0. The plain empirical 90%
    quantile of the same data is 9.1 (numpy's linear interpolation) --
    the two disagree, and the conformal one is deliberately larger.
    """
    residuals = np.arange(1.0, 11.0)
    assert conformal_quantile(residuals, alpha=0.1) == pytest.approx(10.0)
    assert float(np.quantile(np.abs(residuals), 0.9)) == pytest.approx(9.1)


def test_conformal_quantile_uses_absolute_residuals() -> None:
    """Sign must not matter: a residual of -8 is an 8 kW miss."""
    assert conformal_quantile(np.array([-8.0, 1.0, 2.0, 3.0]), alpha=0.5) == pytest.approx(
        conformal_quantile(np.array([8.0, 1.0, 2.0, 3.0]), alpha=0.5)
    )


def test_finite_sample_correction_is_not_cosmetic() -> None:
    """With n = 19 and alpha = 0.1 the required rank is
    ceil(20 * 0.9) = 18, the 18th smallest of 19 -- NOT the 17th that a
    plain `1 - alpha` empirical quantile (index int(0.9 * 19) = 17)
    would take. Dropping the (n + 1) silently narrows the interval.
    """
    residuals = np.arange(1.0, 20.0)
    assert conformal_quantile(residuals, alpha=0.1) == pytest.approx(18.0)
    assert float(np.quantile(residuals, 0.9)) == pytest.approx(17.2)


def test_conformal_quantile_rejects_n_too_small_for_alpha() -> None:
    """n=5 at alpha=0.1 needs rank ceil(6*0.9)=6 > 5. Returning the max
    score instead would look like an answer and carry no guarantee.
    """
    with pytest.raises(ValueError, match="cannot support alpha"):
        conformal_quantile(np.arange(5.0), alpha=0.1)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_conformal_quantile_rejects_alpha_outside_unit_interval(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        conformal_quantile(np.arange(100.0), alpha=alpha)


def test_conformal_quantile_rejects_non_finite_residuals() -> None:
    residuals = np.arange(100.0)
    residuals[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        conformal_quantile(residuals)


def test_conformal_quantile_rejects_empty_residuals() -> None:
    with pytest.raises(ValueError, match="at least one point"):
        conformal_quantile(np.array([]))


def test_conformal_quantile_rejects_mismatched_scale() -> None:
    with pytest.raises(ValueError, match="must match residuals"):
        conformal_quantile(np.arange(100.0), scale=np.ones(50))


def test_conformal_quantile_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        conformal_quantile(np.arange(100.0), scale=np.zeros(100))


# --- the guarantee itself -------------------------------------------------


def test_coverage_reaches_target_on_exchangeable_data() -> None:
    """Property test of the guarantee. On i.i.d. data -- where
    exchangeability genuinely holds -- coverage must land at or above
    1 - alpha. This is the case the theory covers, and it is the
    baseline the real-data failures in reports/08_counterfactual.md are
    measured against.
    """
    rng = np.random.default_rng(SEED)
    prediction = np.full(4000, 1000.0)
    measured = prediction + rng.standard_t(df=3, size=4000) * 200.0

    calibration, scored = time_ordered_split(measured.size, 0.5)
    quantile = conformal_quantile(measured[calibration] - prediction[calibration], DEFAULT_ALPHA)
    interval = conformal_interval(prediction[scored], quantile, alpha=DEFAULT_ALPHA)
    coverage = validate_coverage(measured[scored], interval)

    assert coverage.passed
    assert coverage.empirical_pct == pytest.approx(90.0, abs=2.0)


def test_smaller_alpha_gives_wider_interval() -> None:
    """Monotonicity: asking for more confidence can only cost width."""
    rng = np.random.default_rng(SEED)
    residuals = rng.normal(0.0, 100.0, 2000)
    assert conformal_quantile(residuals, 0.05) > conformal_quantile(residuals, 0.2)


# --- interval construction ------------------------------------------------


def test_interval_is_symmetric_before_clipping() -> None:
    interval = conformal_interval(np.array([500.0, 900.0]), quantile=100.0, lower_clip=None)
    assert interval.lower == pytest.approx([400.0, 800.0])
    assert interval.upper == pytest.approx([600.0, 1000.0])
    assert interval.width == pytest.approx([200.0, 200.0])


def test_lower_clip_enforces_the_physical_floor() -> None:
    """A cooling plant cannot deliver negative cooling, so the lower
    endpoint stops at zero. The upper endpoint is untouched -- clipping
    both would move the interval rather than truncate it.
    """
    interval = conformal_interval(np.array([50.0]), quantile=200.0)
    assert interval.lower == pytest.approx([0.0])
    assert interval.upper == pytest.approx([250.0])


def test_normalised_interval_width_follows_the_prediction() -> None:
    prediction = np.array([1000.0, 5000.0])
    interval = conformal_interval(
        prediction, quantile=0.2, scale=normalising_scale(prediction), lower_clip=None
    )
    assert interval.normalised
    assert interval.width[1] == pytest.approx(5.0 * interval.width[0])


def test_normalising_scale_floors_near_zero_predictions() -> None:
    """Without the floor, an hour predicted at 0 kW would get a
    zero-width interval -- perfect confidence exactly where the inverse
    model's clip destroyed the information.
    """
    scale = normalising_scale(np.array([0.0, 1000.0]), min_fraction=0.05)
    assert scale[0] == pytest.approx(0.05 * 500.0)
    assert scale[1] == pytest.approx(1000.0)


def test_conformal_interval_rejects_negative_quantile() -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        conformal_interval(np.array([1.0]), quantile=-1.0)


def test_validate_coverage_rejects_misaligned_measurements() -> None:
    interval = conformal_interval(np.array([1.0, 2.0]), quantile=1.0)
    with pytest.raises(ValueError, match="does not align"):
        validate_coverage(np.array([1.0]), interval)


def test_coverage_result_reports_relative_width() -> None:
    interval = conformal_interval(np.array([1000.0, 1000.0]), quantile=100.0)
    coverage = validate_coverage(np.array([1000.0, 1000.0]), interval)
    assert coverage.empirical_pct == pytest.approx(100.0)
    assert coverage.mean_relative_width_pct == pytest.approx(20.0)


# --- conditional coverage -------------------------------------------------


def test_coverage_by_group_finds_the_group_a_marginal_number_hides() -> None:
    """Two regimes: 200 quiet hours the model nails, 200 loud hours it
    misses entirely. Marginal coverage is 50%; the per-group split says
    100% and 0%. The whole reason `coverage_by_group` exists.
    """
    prediction = np.full(400, 1000.0)
    measured = np.concatenate([np.full(200, 1000.0), np.full(200, 5000.0)])
    groups = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    interval = conformal_interval(prediction, quantile=100.0)

    assert validate_coverage(measured, interval).empirical_pct == pytest.approx(50.0)
    by_group = coverage_by_group(measured, interval, groups)
    assert by_group[0].empirical_pct == pytest.approx(100.0)
    assert by_group[1].empirical_pct == pytest.approx(0.0)


def test_coverage_by_group_rejects_misaligned_groups() -> None:
    interval = conformal_interval(np.array([1.0, 2.0]), quantile=1.0)
    with pytest.raises(ValueError, match="must align"):
        coverage_by_group(np.array([1.0, 2.0]), interval, np.array([0]))


# --- Mondrian -------------------------------------------------------------


def test_mondrian_gives_the_noisier_group_a_wider_quantile() -> None:
    rng = np.random.default_rng(SEED)
    quiet = rng.normal(0.0, 50.0, 1000)
    loud = rng.normal(0.0, 500.0, 1000)
    residuals = np.concatenate([quiet, loud])
    groups = np.concatenate([np.zeros(1000, dtype=int), np.ones(1000, dtype=int)])

    quantiles, pooled = mondrian_quantiles(residuals, groups, DEFAULT_ALPHA)
    assert quantiles[1] > pooled > quantiles[0]


def test_mondrian_falls_back_for_small_groups() -> None:
    """A group with fewer points than `min_group` takes the pooled
    quantile rather than a quantile of its own that no guarantee covers.
    """
    rng = np.random.default_rng(SEED)
    residuals = np.concatenate([rng.normal(0.0, 50.0, 1000), rng.normal(0.0, 500.0, 20)])
    groups = np.concatenate([np.zeros(1000, dtype=int), np.ones(20, dtype=int)])
    quantiles, pooled = mondrian_quantiles(residuals, groups, DEFAULT_ALPHA, min_group=100)
    assert quantiles[1] == pytest.approx(pooled)


def test_mondrian_interval_uses_pooled_quantile_for_unseen_groups() -> None:
    interval = mondrian_interval(
        np.array([1000.0, 1000.0]),
        np.array([0, 7]),
        {0: 50.0},
        pooled_quantile=300.0,
        lower_clip=None,
    )
    assert interval.width == pytest.approx([100.0, 600.0])


def test_mondrian_interval_rejects_misaligned_groups() -> None:
    with pytest.raises(ValueError, match="must align"):
        mondrian_interval(np.array([1.0, 2.0]), np.array([0]), {}, pooled_quantile=1.0)


# --- splits ---------------------------------------------------------------


def test_time_ordered_split_respects_the_embargo() -> None:
    calibration, scored = time_ordered_split(1000, 0.7, embargo_hours=100)
    assert calibration == slice(0, 700)
    assert scored == slice(800, 1000)


def test_time_ordered_split_rejects_an_embargo_that_eats_the_test_block() -> None:
    with pytest.raises(ValueError, match="leaves one of the two blocks empty"):
        time_ordered_split(1000, 0.7, embargo_hours=400)


def test_interleaved_split_is_disjoint_and_covering() -> None:
    calibration, scored = interleaved_block_split(8760, block_hours=168)
    assert not np.any(calibration & scored)
    assert np.all(calibration | scored)


def test_interleaved_split_spans_the_whole_series_on_both_sides() -> None:
    """The property that makes it worth having: unlike a contiguous
    split, both blocks contain hours from the start AND the end of the
    year, so calibration and scoring see the same seasons.
    """
    calibration, scored = interleaved_block_split(8760, block_hours=168)
    for mask in (calibration, scored):
        indices = np.flatnonzero(mask)
        assert indices.min() < 8760 * 0.1
        assert indices.max() > 8760 * 0.9


def test_interleaved_split_is_deterministic() -> None:
    first = interleaved_block_split(8760)
    second = interleaved_block_split(8760)
    assert np.array_equal(first[0], second[0])


# --- block bootstrap ------------------------------------------------------


def test_block_bootstrap_is_wider_than_the_iid_version_on_correlated_data() -> None:
    """The reason blocks exist. On a strongly autocorrelated series an
    hour-by-hour bootstrap reports a far tighter interval for the mean
    than the data supports -- narrow, centred correctly, and wrong.
    """
    hours = np.arange(8760)
    series = 100.0 + 50.0 * np.sin(2.0 * np.pi * hours / 8760.0)
    block_low, block_high = block_bootstrap_ci(series, block_hours=168, n_resamples=400)
    iid_low, iid_high = block_bootstrap_ci(series, block_hours=1, n_resamples=400)
    assert (block_high - block_low) > 3.0 * (iid_high - iid_low)


def test_block_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    rng = np.random.default_rng(SEED)
    series = rng.normal(0.0, 1.0, 2000)
    assert block_bootstrap_ci(series, n_resamples=200) == block_bootstrap_ci(
        series, n_resamples=200
    )


def test_block_bootstrap_brackets_the_sample_mean() -> None:
    rng = np.random.default_rng(SEED)
    series = rng.normal(10.0, 1.0, 4000)
    low, high = block_bootstrap_ci(series, block_hours=168, n_resamples=400)
    assert low < float(series.mean()) < high


def test_block_bootstrap_rejects_a_series_shorter_than_one_block() -> None:
    with pytest.raises(ValueError, match="shorter than one"):
        block_bootstrap_ci(np.arange(50.0), block_hours=168)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"alpha": 0.0}, "alpha must be in"),
        ({"block_hours": 0}, "block_hours must be"),
        ({"n_resamples": 0}, "n_resamples must be"),
    ],
)
def test_block_bootstrap_input_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        block_bootstrap_ci(np.arange(1000.0), **kwargs)


def test_conformal_interval_dataclass_reports_its_target() -> None:
    interval = ConformalInterval(
        lower=np.array([0.0]),
        upper=np.array([1.0]),
        prediction=np.array([0.5]),
        quantile=0.5,
        alpha=0.1,
        n_calibration=100,
    )
    assert interval.target_coverage_pct == pytest.approx(90.0)

"""Tests for the residual decomposition (L7.1).

Every case here is built so the correct answer is known BEFORE the code
runs -- a residual that is exactly a step, exactly a slope, or exactly
nothing. The L6.2 pattern: no reference implementation is consulted,
because a reference implementation can share the bug.

The most important test in the file is
`test_binning_on_measured_load_manufactures_a_slope`. It does not check
that the code is right; it checks that the design decision behind the
code is necessary, by running the same data both ways and showing that
the rejected option invents structure a perfect model does not have.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from cooling_twin import set_seed
from cooling_twin.analysis.residual import (
    MIN_BIN_COUNT,
    MIN_HOURS,
    MIN_STRUCTURE_RATIO,
    Binning,
    ResidualProfile,
    decompose_residual,
    residual_profile,
)

HOURS = 24 * 60  # 60 days -- comfortably above MIN_HOURS


def _index(hours: int = HOURS) -> pd.DatetimeIndex:
    """An hourly, timezone-aware index of the given length."""
    return pd.date_range("2016-01-01", periods=hours, freq="h", tz="UTC")


def _drivers(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Smooth synthetic weather aligned with an index."""
    hour = index.hour.to_numpy(dtype=float)
    day = index.dayofyear.to_numpy(dtype=float)
    return {
        "t_ambient_c": 20.0 + 8.0 * np.sin(2.0 * np.pi * hour / 24.0) + day / 30.0,
        "humidity_ratio_kg_per_kg": 0.008 + 0.002 * np.sin(2.0 * np.pi * day / 365.0),
    }


# ---------------------------------------------------------------------
# sign convention
# ---------------------------------------------------------------------


def test_positive_residual_means_the_model_under_predicts() -> None:
    """G14's convention: measured - predicted, matching nmbe()'s sign.

    Its own test because every other test in this file would pass
    unchanged with the sign flipped -- the profiles would simply mirror.
    """
    index = _index()
    measured = np.full(len(index), 1000.0)
    predicted = np.full(len(index), 900.0)

    decomposition = decompose_residual(
        index, measured, predicted, label="under-predicting", **_drivers(index)
    )

    assert decomposition.mean_residual_kw == pytest.approx(100.0)


# ---------------------------------------------------------------------
# known-answer profiles
# ---------------------------------------------------------------------


def test_a_pure_hour_of_day_step_is_recovered_exactly() -> None:
    """Residual = 500 kW between 09:00 and 17:00, 0 otherwise, no noise."""
    index = _index()
    hour = index.hour.to_numpy(dtype=float)
    residual = np.where((hour >= 9.0) & (hour < 17.0), 500.0, 0.0)

    profile = residual_profile(
        residual,
        hour,
        name="hour_of_day",
        unit="0-23",
        binning=Binning.CATEGORICAL,
        normaliser_kw=1000.0,
    )

    assert profile.counts.size == 24
    assert profile.means[9:17] == pytest.approx(500.0)
    assert profile.means[:9] == pytest.approx(0.0)
    assert profile.means[17:] == pytest.approx(0.0)
    # Noiseless and perfectly aligned with the bins: the binned means
    # account for ALL of the residual's variance.
    assert profile.explained_fraction == pytest.approx(1.0)
    assert profile.swing_kw == pytest.approx(500.0)
    assert profile.swing_pct_of_mean_load == pytest.approx(50.0)
    assert profile.structured


def test_a_constant_residual_has_no_structure_anywhere() -> None:
    """A pure offset is a bias, not a shape. Every driver must read 0.

    This is the 0/0 case -- there is no residual variance for any driver
    to explain -- and it must return 0.0, not nan. A nan would propagate
    into the summary table and read as a missing measurement rather than
    as the finding it is.
    """
    index = _index()
    measured = np.full(len(index), 2000.0)
    predicted = np.full(len(index), 1750.0)

    decomposition = decompose_residual(index, measured, predicted, **_drivers(index))

    assert decomposition.mean_residual_kw == pytest.approx(250.0)
    for profile in decomposition.profiles.values():
        assert profile.explained_fraction == 0.0
        assert profile.swing_kw == pytest.approx(0.0)
        assert not profile.structured
    assert decomposition.structured_drivers == ()


def test_a_temperature_slope_produces_monotonic_bin_means() -> None:
    """Residual = 10 kW/K exactly. Bin means must rise with the bins."""
    index = _index()
    t_ambient = _drivers(index)["t_ambient_c"]
    residual = 10.0 * t_ambient

    profile = residual_profile(
        residual,
        t_ambient,
        name="outdoor_dry_bulb",
        unit="degC",
        binning=Binning.FIXED_WIDTH,
        normaliser_kw=1000.0,
        width=2.5,
    )

    assert np.all(np.diff(profile.means) > 0.0)
    # Not 1.0: within a 2.5 K bin the residual still spans 25 kW, so
    # binned means cannot account for quite everything. That gap is the
    # bin width, not a defect.
    assert profile.explained_fraction > 0.98
    assert np.all(np.diff(profile.centres) > 0.0)


def test_bin_centres_are_the_data_mean_not_the_nominal_centre() -> None:
    """Centres report where the hours actually are inside each bin."""
    index = _index()
    # Everything sits in the top of the 20.0-22.5 bin.
    t_ambient = np.full(len(index), 22.4)
    t_ambient[: len(index) // 2] = 22.3
    residual = np.linspace(-10.0, 10.0, len(index))

    profile = residual_profile(
        residual,
        t_ambient,
        name="outdoor_dry_bulb",
        unit="degC",
        binning=Binning.FIXED_WIDTH,
        normaliser_kw=1000.0,
        width=2.5,
    )

    assert profile.counts.size == 1
    assert profile.centres[0] == pytest.approx(22.35)  # not 21.25


# ---------------------------------------------------------------------
# the design decision this module exists to get right
# ---------------------------------------------------------------------


def test_binning_on_measured_load_manufactures_a_slope() -> None:
    """A PERFECT model, scored both ways, on identical data.

    The model here reproduces the true load exactly; the only error is
    measurement noise. Binned on predicted load the profile is flat,
    which is the truth. Binned on measured load it slopes hard, because
    the noise that put an hour in a high bin is the same noise that
    makes its residual positive. This is why `decompose_residual` never
    offers measured load as a driver.
    """
    rng = set_seed()
    n = 8760
    true_load = rng.uniform(500.0, 5000.0, n)
    noise = rng.normal(0.0, 300.0, n)
    measured = true_load + noise
    predicted = true_load  # a perfect model of the signal
    residual = measured - predicted  # == noise, structureless by construction

    honest = residual_profile(
        residual,
        predicted,
        name="predicted_load",
        unit="kW",
        binning=Binning.QUANTILE,
        normaliser_kw=float(measured.mean()),
    )
    misleading = residual_profile(
        residual,
        measured,
        name="measured_load",
        unit="kW",
        binning=Binning.QUANTILE,
        normaliser_kw=float(measured.mean()),
    )

    assert not honest.structured
    assert honest.structure_ratio < MIN_STRUCTURE_RATIO
    assert honest.swing_kw < 100.0

    assert misleading.structured
    assert misleading.structure_ratio > 20.0 * honest.structure_ratio
    # A ~380 kW swing, rising monotonically end to end, conjured from a
    # model whose only error is the meter's own noise.
    assert misleading.swing_kw > 300.0
    assert misleading.swing_kw > 5.0 * honest.swing_kw
    assert misleading.means[-1] - misleading.means[0] > 300.0


def test_noise_floor_is_reported_and_matches_its_formula() -> None:
    """`(k - 1) / (n - 1)`, and a pure-noise residual lands near it."""
    rng = set_seed()
    n = 8760
    residual = rng.normal(0.0, 200.0, n)
    driver = rng.uniform(0.0, 100.0, n)

    profile = residual_profile(
        residual,
        driver,
        name="driver",
        unit="-",
        binning=Binning.QUANTILE,
        normaliser_kw=1000.0,
        n_bins=10,
    )

    k = profile.counts.size
    assert profile.noise_floor == pytest.approx((k - 1) / (n - 1))
    # Pure noise: the explained share should sit within a small multiple
    # of the floor. That is precisely what MIN_STRUCTURE_RATIO screens.
    assert profile.explained_fraction < MIN_STRUCTURE_RATIO * profile.noise_floor
    assert not profile.structured


# ---------------------------------------------------------------------
# binning behaviour
# ---------------------------------------------------------------------


def test_sparse_bins_are_dropped_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Bins under MIN_BIN_COUNT hours never reach the profile."""
    n = 2000
    driver = np.zeros(n)
    driver[:5] = 100.0  # 5 hours, far away, in a bin of their own
    residual = np.linspace(-5.0, 5.0, n)

    with caplog.at_level(logging.INFO, logger="cooling_twin.analysis.residual"):
        profile = residual_profile(
            residual,
            driver,
            name="driver",
            unit="-",
            binning=Binning.FIXED_WIDTH,
            normaliser_kw=1000.0,
            width=2.5,
        )

    assert profile.counts.size == 1
    assert int(profile.counts[0]) == n - 5
    assert np.all(profile.counts >= MIN_BIN_COUNT)
    assert "dropped 1 of 2 bins" in caplog.text


def test_quantile_bins_hold_comparable_counts_on_a_skewed_driver() -> None:
    """The reason quantile binning exists: a fixed-width tail is empty."""
    rng = set_seed()
    n = 8760
    driver = rng.exponential(500.0, n)  # heavily right-skewed, like load
    residual = rng.normal(0.0, 100.0, n)

    profile = residual_profile(
        residual,
        driver,
        name="driver",
        unit="kW",
        binning=Binning.QUANTILE,
        normaliser_kw=1000.0,
        n_bins=10,
    )

    assert profile.counts.size == 10
    assert profile.counts.max() - profile.counts.min() <= 1


def test_categorical_binning_rejects_non_integer_drivers() -> None:
    """Hour-of-day binning applied to a continuous driver is a bug."""
    with pytest.raises(ValueError, match="integer-valued"):
        residual_profile(
            np.zeros(100),
            np.linspace(0.0, 10.5, 100),
            name="driver",
            unit="-",
            binning=Binning.CATEGORICAL,
            normaliser_kw=1000.0,
        )


def test_a_constant_driver_explains_nothing_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One bin, zero explained, a warning -- not an exception.

    The L6.4 annual-mean baseline predicts a single constant, and its
    residual is worth profiling. Raising here would make a legitimate
    decomposition fail on the one driver that happens to be flat.
    """
    with caplog.at_level(logging.WARNING, logger="cooling_twin.analysis.residual"):
        profile = residual_profile(
            np.linspace(0.0, 1.0, 100),
            np.full(100, 7.0),
            name="driver",
            unit="-",
            binning=Binning.QUANTILE,
            normaliser_kw=1000.0,
        )

    assert profile.counts.size == 1
    assert profile.explained_fraction == 0.0
    assert profile.noise_floor == 0.0
    assert profile.structure_ratio == 0.0
    assert not profile.structured
    assert "collapsed to a single bin" in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"binning": Binning.FIXED_WIDTH, "width": 0.0}, "width > 0"),
        ({"binning": Binning.QUANTILE, "n_bins": 1}, "n_bins >= 2"),
    ],
)
def test_invalid_binning_arguments_raise(kwargs: dict, match: str) -> None:
    """Bad bin geometry fails loudly rather than producing one bin."""
    with pytest.raises(ValueError, match=match):
        residual_profile(
            np.linspace(0.0, 1.0, 100),
            np.linspace(0.0, 100.0, 100),
            name="driver",
            unit="-",
            normaliser_kw=1000.0,
            **kwargs,
        )


def test_every_bin_too_thin_raises() -> None:
    """No usable bin is an error, not an empty profile."""
    with pytest.raises(ValueError, match="no bin of driver holds"):
        residual_profile(
            np.linspace(0.0, 1.0, 30),
            np.linspace(0.0, 1000.0, 30),
            name="driver",
            unit="-",
            binning=Binning.FIXED_WIDTH,
            normaliser_kw=1000.0,
            width=2.5,
        )


# ---------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------


def test_misaligned_series_raise() -> None:
    """Length mismatch is caught before it becomes a broadcast."""
    with pytest.raises(ValueError, match="align element-wise"):
        residual_profile(
            np.zeros(100),
            np.zeros(99),
            name="driver",
            unit="-",
            binning=Binning.QUANTILE,
            normaliser_kw=1000.0,
        )


def test_non_finite_values_raise() -> None:
    """NaN removal belongs to M3's cleaning pipeline, not here."""
    residual = np.zeros(100)
    residual[7] = np.nan
    with pytest.raises(ValueError, match="must both be finite"):
        residual_profile(
            residual,
            np.linspace(0.0, 100.0, 100),
            name="driver",
            unit="-",
            binning=Binning.QUANTILE,
            normaliser_kw=1000.0,
        )


def test_empty_residual_raises() -> None:
    """An empty profile would report `structured=False` for no data."""
    with pytest.raises(ValueError, match="empty residual"):
        residual_profile(
            np.array([]),
            np.array([]),
            name="driver",
            unit="-",
            binning=Binning.QUANTILE,
            normaliser_kw=1000.0,
        )


@pytest.mark.parametrize("normaliser", [0.0, -100.0, float("nan")])
def test_non_positive_normaliser_raises(normaliser: float) -> None:
    """Swing percentages are undefined without a positive mean load."""
    with pytest.raises(ValueError, match="positive mean load"):
        residual_profile(
            np.linspace(0.0, 1.0, 100),
            np.linspace(0.0, 100.0, 100),
            name="driver",
            unit="-",
            binning=Binning.QUANTILE,
            normaliser_kw=normaliser,
        )


def test_decompose_rejects_a_non_datetime_index() -> None:
    """Hour-of-day and month can only come from real timestamps."""
    index = pd.RangeIndex(HOURS)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        decompose_residual(
            index,  # type: ignore[arg-type]
            np.full(HOURS, 1000.0),
            np.full(HOURS, 900.0),
            t_ambient_c=np.full(HOURS, 25.0),
            humidity_ratio_kg_per_kg=np.full(HOURS, 0.008),
        )


def test_decompose_rejects_too_short_a_window() -> None:
    """A week of data cannot fill 24 hour-of-day bins meaningfully."""
    hours = MIN_HOURS - 1
    index = _index(hours)
    with pytest.raises(ValueError, match="too few to profile"):
        decompose_residual(
            index,
            np.full(hours, 1000.0),
            np.full(hours, 900.0),
            **_drivers(index),
        )


def test_decompose_rejects_misaligned_series() -> None:
    """One short array must not be broadcast against the index."""
    index = _index()
    with pytest.raises(ValueError, match="same length"):
        decompose_residual(
            index,
            np.full(len(index), 1000.0),
            np.full(len(index) - 1, 900.0),
            **_drivers(index),
        )


def test_decompose_rejects_non_finite_series() -> None:
    """A single NaN in the model output must not reach the profiles."""
    index = _index()
    predicted = np.full(len(index), 900.0)
    predicted[3] = np.inf
    with pytest.raises(ValueError, match="must both be finite"):
        decompose_residual(
            index, np.full(len(index), 1000.0), predicted, **_drivers(index)
        )


def test_an_empty_profile_reports_zero_swing() -> None:
    """`ResidualProfile` is public, so its properties must not divide by
    an empty array if one is constructed directly."""
    empty = ResidualProfile(
        driver="driver",
        unit="-",
        centres=np.array([]),
        counts=np.array([], dtype=np.int64),
        means=np.array([]),
        sems=np.array([]),
        explained_fraction=0.0,
        noise_floor=0.0,
        normaliser_kw=1000.0,
    )

    assert empty.swing_kw == 0.0
    assert empty.swing_pct_of_mean_load == 0.0
    assert not empty.structured


def test_decompose_rejects_non_positive_mean_load() -> None:
    """Normalising a swing by a zero mean is undefined."""
    index = _index()
    with pytest.raises(ValueError, match="mean measured load"):
        decompose_residual(
            index,
            np.zeros(len(index)),
            np.full(len(index), 900.0),
            **_drivers(index),
        )


# ---------------------------------------------------------------------
# the decomposition object
# ---------------------------------------------------------------------


def test_decomposition_covers_the_five_standard_drivers() -> None:
    """The report table's rows are fixed, so a missing one is a defect."""
    index = _index()
    drivers = _drivers(index)
    measured = 2000.0 + 30.0 * drivers["t_ambient_c"]

    decomposition = decompose_residual(
        index, measured, measured * 0.9, label="check", **drivers
    )

    assert set(decomposition.profiles) == {
        "month",
        "hour_of_day",
        "predicted_load",
        "outdoor_dry_bulb",
        "humidity_ratio",
    }
    assert decomposition.label == "check"
    assert decomposition.residual_kw.size == len(index)


def test_summary_is_ordered_by_structure_ratio() -> None:
    """The most structured driver is the one to read first."""
    index = _index()
    drivers = _drivers(index)
    hour = index.hour.to_numpy(dtype=float)
    measured = 3000.0 + np.where((hour >= 8.0) & (hour < 18.0), 600.0, 0.0)
    predicted = np.full(len(index), 3000.0)

    decomposition = decompose_residual(index, measured, predicted, **drivers)
    summary = decomposition.summary()

    assert list(summary["ratio"]) == sorted(summary["ratio"], reverse=True)
    assert summary.index[0] == "hour_of_day"
    assert decomposition.structured_drivers[0] == "hour_of_day"


def test_profiles_and_decomposition_are_immutable() -> None:
    """A number cannot change between computation and the report."""
    index = _index()
    decomposition = decompose_residual(
        index,
        np.full(len(index), 1000.0),
        np.full(len(index), 900.0),
        **_drivers(index),
    )

    with pytest.raises(AttributeError):
        decomposition.label = "edited"  # type: ignore[misc]
    with pytest.raises(TypeError):
        decomposition.profiles["month"] = decomposition.profiles["hour_of_day"]  # type: ignore[index]
    with pytest.raises(AttributeError):
        decomposition.profiles["month"].explained_fraction = 0.99  # type: ignore[misc]


def test_profile_to_frame_round_trips_the_numbers() -> None:
    """The table a report prints must be the profile's own numbers."""
    index = _index()
    hour = index.hour.to_numpy(dtype=float)
    residual = np.where(hour < 12.0, 100.0, -100.0)

    profile: ResidualProfile = residual_profile(
        residual,
        hour,
        name="hour_of_day",
        unit="0-23",
        binning=Binning.CATEGORICAL,
        normaliser_kw=1000.0,
    )
    frame = profile.to_frame()

    assert list(frame.index) == list(profile.centres)
    assert frame["mean residual kW"].to_numpy() == pytest.approx(profile.means)
    assert int(frame["hours"].sum()) == len(index)

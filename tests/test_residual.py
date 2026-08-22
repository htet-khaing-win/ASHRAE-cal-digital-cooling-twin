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
    autocorrelation,
    band_edges_from_quantiles,
    decompose_residual,
    effective_sample_size,
    fit_residual_curvature,
    linear_residual_slopes,
    matched_band_split,
    residual_diagnostics,
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


# ---------------------------------------------------------------------
# slopes -- the profile's finding, in the units a fix is written in
# ---------------------------------------------------------------------


def test_slopes_recover_an_exactly_known_plane() -> None:
    """Residual built to lie exactly on 200 + 40*T + 15*w_g. No noise.

    Known-answer, in the L6.2 style: OLS on data that lies exactly on a
    plane has an exact solution, so a wrong design matrix (missing
    intercept, swapped columns, humidity left in kg/kg) cannot hide
    inside a tolerance.
    """
    rng = set_seed()
    n = 2000
    temperature = rng.uniform(5.0, 45.0, n)
    humidity_g = rng.uniform(2.0, 18.0, n)
    residual = 200.0 + 40.0 * temperature + 15.0 * humidity_g

    slopes = linear_residual_slopes(residual, temperature, humidity_g / 1000.0)

    assert slopes["intercept_kw"] == pytest.approx(200.0, abs=1e-6)
    assert slopes["slope_kw_per_K"] == pytest.approx(40.0, abs=1e-9)
    assert slopes["slope_kw_per_g_per_kg"] == pytest.approx(15.0, abs=1e-9)
    assert slopes["mean_residual_kw"] == pytest.approx(float(residual.mean()))


def test_slopes_are_joint_not_two_univariate_fits() -> None:
    """With correlated drivers, the joint fit must recover the truth.

    Humidity here is a deterministic function of temperature plus a
    little spread. A univariate slope on temperature would absorb the
    humidity term and read far above 40; the multiple regression must
    not. This is why the two slopes come from one design matrix.
    """
    rng = set_seed()
    n = 5000
    temperature = rng.uniform(5.0, 45.0, n)
    humidity_g = 0.3 * temperature + rng.uniform(-1.0, 1.0, n)
    residual = 40.0 * temperature + 15.0 * humidity_g

    slopes = linear_residual_slopes(residual, temperature, humidity_g / 1000.0)
    univariate = float(
        np.polyfit(temperature, residual, 1)[0]
    )  # the mistake being ruled out

    assert slopes["slope_kw_per_K"] == pytest.approx(40.0, abs=1e-6)
    assert slopes["slope_kw_per_g_per_kg"] == pytest.approx(15.0, abs=1e-6)
    assert univariate > 44.0  # carries 0.3 * 15 = 4.5 kW/K of humidity


def test_slopes_report_correlations_beside_the_coefficients() -> None:
    """Correlation is scale-free, so g/kg and kg/kg must agree."""
    rng = set_seed()
    n = 1000
    temperature = rng.uniform(5.0, 45.0, n)
    humidity = rng.uniform(0.002, 0.018, n)
    residual = 30.0 * temperature + rng.normal(0.0, 50.0, n)

    slopes = linear_residual_slopes(residual, temperature, humidity)

    assert slopes["corr_temperature"] == pytest.approx(
        float(np.corrcoef(residual, temperature)[0, 1])
    )
    assert slopes["corr_humidity"] == pytest.approx(
        float(np.corrcoef(residual, humidity)[0, 1])
    )


def test_slopes_reject_misaligned_or_non_finite_input() -> None:
    """Same validation as the profiles -- one helper, one set of rules."""
    with pytest.raises(ValueError, match="align element-wise"):
        linear_residual_slopes(np.zeros(100), np.zeros(99), np.zeros(100))
    with pytest.raises(ValueError, match="must both be finite"):
        linear_residual_slopes(
            np.zeros(100), np.zeros(100), np.full(100, np.nan)
        )


# ---------------------------------------------------------------------
# curvature -- the U-shape found on real data at L7.1b
# ---------------------------------------------------------------------


def _parabola(n: int = 6000, curvature: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """A residual lying exactly on `curvature * (T - 20)^2 - 400`."""
    rng = set_seed()
    temperature = rng.uniform(0.0, 40.0, n)
    residual = curvature * (temperature - 20.0) ** 2 - 400.0
    return residual, temperature


def test_a_known_parabola_is_recovered_exactly() -> None:
    """Known-answer: the quadratic coefficient and the vertex."""
    residual, temperature = _parabola(curvature=2.0)

    fit = fit_residual_curvature(residual, temperature, driver="outdoor_dry_bulb")

    assert fit.quadratic_kw_per_unit2 == pytest.approx(2.0, abs=1e-9)
    assert fit.vertex == pytest.approx(20.0, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.is_u_shaped


def test_the_quadratic_term_is_invariant_to_centring() -> None:
    """Centring changes the basis, not the curve.

    The implementation centres the driver to keep the design matrix well
    conditioned. That is only legitimate if the reported second-order
    coefficient is unchanged by it, so the claim is tested rather than
    asserted: shifting the driver by 100 units must not move it.
    """
    residual, temperature = _parabola(curvature=1.5)

    here = fit_residual_curvature(residual, temperature, driver="d")
    shifted = fit_residual_curvature(residual, temperature + 100.0, driver="d")

    assert shifted.quadratic_kw_per_unit2 == pytest.approx(
        here.quadratic_kw_per_unit2, rel=1e-9
    )
    assert shifted.vertex == pytest.approx(here.vertex + 100.0, rel=1e-9)


def test_a_straight_line_is_not_called_u_shaped() -> None:
    """The failure mode this verdict exists to avoid.

    A pure slope has one band above the middle and one below. Calling
    that a U would turn every ordinary bias into a structural finding.
    """
    rng = set_seed()
    temperature = rng.uniform(0.0, 40.0, 6000)
    residual = 30.0 * temperature + rng.normal(0.0, 50.0, 6000)

    fit = fit_residual_curvature(residual, temperature, driver="d")

    assert not fit.is_u_shaped
    assert fit.low_lift_kw < 0.0  # the cold band sits BELOW the middle
    assert fit.high_lift_kw > 0.0
    assert fit.r_squared == pytest.approx(fit.linear_r_squared, abs=1e-3)


def test_an_inverted_u_is_not_called_u_shaped() -> None:
    """Concave curvature is a real finding, but it is a different one."""
    residual, temperature = _parabola(curvature=-2.0)

    fit = fit_residual_curvature(residual, temperature, driver="d")

    assert fit.quadratic_kw_per_unit2 == pytest.approx(-2.0, abs=1e-9)
    assert not fit.is_u_shaped
    assert fit.low_lift_kw < 0.0
    assert fit.high_lift_kw < 0.0


def test_noise_alone_is_not_called_u_shaped() -> None:
    """Both arms must clear their combined standard error."""
    rng = set_seed()
    temperature = rng.uniform(0.0, 40.0, 8760)
    residual = rng.normal(0.0, 300.0, 8760)

    fit = fit_residual_curvature(residual, temperature, driver="d")

    assert not fit.is_u_shaped
    assert abs(fit.r_squared) < 0.01


def test_curvature_beats_a_straight_line_on_a_parabola() -> None:
    """The gain from the quadratic term is reported, not assumed."""
    residual, temperature = _parabola(curvature=2.0)

    fit = fit_residual_curvature(residual, temperature, driver="d")

    assert fit.r_squared > 0.99
    # A straight line through a symmetric parabola explains almost
    # nothing -- the two arms cancel, exactly as they do on real data.
    assert fit.linear_r_squared < 0.05


def test_band_edges_are_terciles_and_reusable_across_years() -> None:
    """Edges derived once, then applied to a different distribution."""
    rng = set_seed()
    train = rng.uniform(0.0, 30.0, 8760)
    edges = band_edges_from_quantiles(train)

    assert edges[0] == pytest.approx(10.0, abs=0.5)
    assert edges[1] == pytest.approx(20.0, abs=0.5)

    # A warmer year, scored on the TRAINING year's bands: the band counts
    # must now be uneven, which is the point -- the bands did not move.
    warmer = train + 5.0
    fit = fit_residual_curvature(
        rng.normal(0.0, 10.0, 8760), warmer, driver="d", band_edges=edges
    )

    assert fit.band_edges == edges
    assert fit.band_counts[0] < fit.band_counts[2]
    assert sum(fit.band_counts) == 8760


@pytest.mark.parametrize("bad", [0.0, 0.5, 0.9, -0.1])
def test_band_quantile_outside_the_open_interval_raises(bad: float) -> None:
    """A quantile of 0.5 leaves no middle band at all."""
    with pytest.raises(ValueError, match=r"quantile must lie in \(0, 0.5\)"):
        band_edges_from_quantiles(np.linspace(0.0, 40.0, 1000), quantile=bad)


def test_bands_from_a_non_overlapping_year_raise() -> None:
    """A band that catches no hours is an error, not an empty mean."""
    rng = set_seed()
    with pytest.raises(ValueError, match="band holds only"):
        fit_residual_curvature(
            rng.normal(0.0, 10.0, 1000),
            rng.uniform(30.0, 40.0, 1000),
            driver="d",
            band_edges=(0.0, 10.0),
        )


def test_a_constant_driver_cannot_be_banded() -> None:
    """Coinciding edges would put every hour in one band."""
    with pytest.raises(ValueError, match="band edges coincide"):
        band_edges_from_quantiles(np.full(1000, 20.0))


# ---------------------------------------------------------------------
# L7.2 -- is the residual random?
# ---------------------------------------------------------------------


def test_autocorrelation_of_white_noise_is_near_zero() -> None:
    """The null case, and the scale everything else is read against."""
    rng = set_seed()
    residual = rng.normal(0.0, 100.0, 20000)

    acf = autocorrelation(residual, lags=(1, 24, 168))

    for value in acf.values():
        assert abs(value) < 0.05


def test_autocorrelation_of_a_pure_sine_is_known_by_construction() -> None:
    """A 24-hour cycle, against the exact value of the BIASED estimator.

    For a pure sinusoid the estimator this module uses returns
    `cos(2*pi*k/period) * (1 - k/n)` exactly -- the cosine from the
    signal, the `(1 - k/n)` from dividing by `n` instead of by the
    `n - k` products actually summed. Pinning the shrinkage rather than
    tolerating it is deliberate: an unbiased estimator would return the
    cosine alone, pass a loose "approximately 1.0" assertion, and
    silently disagree with the Ljung-Box statistic computed from it.
    """
    hours = np.arange(24 * 400, dtype=float)
    residual = np.sin(2.0 * np.pi * hours / 24.0)
    n = residual.size

    acf = autocorrelation(residual, lags=(6, 12, 24))

    assert acf[24] == pytest.approx(1.0 * (1 - 24 / n), abs=1e-6)
    assert acf[12] == pytest.approx(-1.0 * (1 - 12 / n), abs=1e-6)
    assert acf[6] == pytest.approx(0.0, abs=1e-3)


def test_autocorrelation_of_an_ar1_process_recovers_its_coefficient() -> None:
    """An AR(1) with phi = 0.9 must read rho(1) ~ 0.9 and rho(2) ~ 0.81."""
    rng = set_seed()
    n, phi = 200000, 0.9
    noise = rng.normal(0.0, 1.0, n)
    residual = np.zeros(n)
    for index in range(1, n):
        residual[index] = phi * residual[index - 1] + noise[index]

    acf = autocorrelation(residual[1000:], lags=(1, 2, 3))

    assert acf[1] == pytest.approx(phi, abs=0.01)
    assert acf[2] == pytest.approx(phi**2, abs=0.02)
    assert acf[3] == pytest.approx(phi**3, abs=0.03)


def test_white_noise_loses_variance_to_daily_averaging() -> None:
    """The 1/24 null: averaging 24 independent draws cuts variance 24x."""
    rng = set_seed()
    diagnostics = residual_diagnostics(rng.normal(0.0, 100.0, 24 * 400))

    assert diagnostics.daily_variance_share == pytest.approx(1.0 / 24.0, abs=0.01)
    assert diagnostics.white_noise_variance_share == pytest.approx(1.0 / 24.0)
    assert not diagnostics.survives_daily_averaging


def test_a_slow_drift_survives_daily_averaging() -> None:
    """The signature that says the remaining error is learnable."""
    rng = set_seed()
    hours = np.arange(24 * 400, dtype=float)
    seasonal = 500.0 * np.sin(2.0 * np.pi * hours / (24.0 * 365.0))
    diagnostics = residual_diagnostics(seasonal + rng.normal(0.0, 50.0, hours.size))

    assert diagnostics.daily_variance_share > 0.9
    assert diagnostics.survives_daily_averaging


def test_ljung_box_does_not_reject_white_noise() -> None:
    """The test must be capable of NOT firing, or it measures nothing."""
    rng = set_seed()
    diagnostics = residual_diagnostics(
        rng.normal(0.0, 100.0, 20000), ljung_box_lags=24
    )

    assert diagnostics.ljung_box_p > 0.01
    assert diagnostics.ljung_box_lags == 24


def test_ljung_box_rejects_an_autocorrelated_series() -> None:
    """And it must fire on structure, with the statistic far above df."""
    rng = set_seed()
    n = 20000
    noise = rng.normal(0.0, 1.0, n)
    residual = np.zeros(n)
    for index in range(1, n):
        residual[index] = 0.8 * residual[index - 1] + noise[index]

    diagnostics = residual_diagnostics(residual, ljung_box_lags=24)

    assert diagnostics.ljung_box_p < 1e-10
    assert diagnostics.ljung_box_q > 100.0 * diagnostics.ljung_box_lags


def test_diagnostics_reject_invalid_lags() -> None:
    """A lag longer than the series would silently compare nothing."""
    rng = set_seed()
    residual = rng.normal(0.0, 1.0, 500)
    with pytest.raises(ValueError, match="shorter than the series"):
        residual_diagnostics(residual, ljung_box_lags=500)
    with pytest.raises(ValueError, match="shorter than the series"):
        autocorrelation(residual, lags=(1, 600))
    with pytest.raises(ValueError, match="at least 1"):
        autocorrelation(residual, lags=(0,))


def test_a_constant_residual_has_undefined_autocorrelation() -> None:
    """Zero variance is a bias, and the profiles are the tool for it."""
    with pytest.raises(ValueError, match="zero variance"):
        autocorrelation(np.full(1000, 42.0))


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (np.array([]), "empty driver"),
        (np.array([1.0, np.nan, 3.0]), "driver must be finite"),
    ],
)
def test_band_edges_reject_unusable_drivers(values: np.ndarray, match: str) -> None:
    """Bands derived from missing data would move silently between runs."""
    with pytest.raises(ValueError, match=match):
        band_edges_from_quantiles(values)


@pytest.mark.parametrize(
    ("series", "match"),
    [
        (np.array([]), "non-empty one-dimensional"),
        (np.zeros((10, 2)), "non-empty one-dimensional"),
        (np.array([1.0, np.nan, 3.0, 4.0]), "residual must be finite"),
    ],
)
def test_autocorrelation_rejects_unusable_series(
    series: np.ndarray, match: str
) -> None:
    """A NaN would poison every lag; a 2-D array would lag the wrong axis."""
    with pytest.raises(ValueError, match=match):
        autocorrelation(series, lags=(1,))


# ---------------------------------------------------------------------
# L7.2 -- n is not n when the data is autocorrelated
# ---------------------------------------------------------------------


def test_white_noise_is_worth_its_full_sample_size() -> None:
    """The null: independent draws lose nothing."""
    rng = set_seed()
    n = 40000
    assert effective_sample_size(rng.normal(0.0, 1.0, n)) == pytest.approx(
        n, rel=0.1
    )


def test_ar1_effective_size_matches_its_closed_form() -> None:
    """Known-answer: for AR(1), the inflation is (1+phi)/(1-phi).

    phi = 0.5 gives an inflation of exactly 3, so 60,000 correlated
    samples are worth 20,000 independent ones. Derivable on paper --
    sum_k phi^k = phi/(1-phi), and 1 + 2*phi/(1-phi) = (1+phi)/(1-phi).
    """
    rng = set_seed()
    n, phi = 60000, 0.5
    noise = rng.normal(0.0, 1.0, n)
    series = np.zeros(n)
    for index in range(1, n):
        series[index] = phi * series[index - 1] + noise[index]

    n_eff = effective_sample_size(series[1000:], max_lag=200)

    assert n_eff == pytest.approx((n - 1000) * (1 - phi) / (1 + phi), rel=0.2)


def test_long_memory_costs_far_more_than_ar1_would_suggest() -> None:
    """The case that motivated the estimator choice.

    A slow cycle stays correlated for hundreds of lags. The AR(1) form
    reads only rho(1) and would report a mild correction; the summed
    form sees the whole tail. This is Claude's residual in miniature.
    """
    rng = set_seed()
    hours = np.arange(24 * 365, dtype=float)
    # Heavy noise on top of a slow cycle: the noise drags rho(1) down to
    # an unremarkable value while the cycle keeps the tail positive for
    # hundreds of lags. That combination -- modest rho(1), long memory --
    # is where reading only the first lag goes most badly wrong, and it
    # is the shape of every residual measured on this project.
    series = 100.0 * np.sin(2.0 * np.pi * hours / (24.0 * 60.0)) + rng.normal(
        0.0, 60.0, hours.size
    )

    acf = autocorrelation(series, (1, 168))
    ar1_estimate = series.size * (1 - acf[1]) / (1 + acf[1])
    n_eff = effective_sample_size(series)

    assert acf[1] < 0.6  # rho(1) alone looks mild
    assert acf[168] > 0.3  # but a week later it is still strongly correlated
    assert ar1_estimate > 10.0 * n_eff  # so AR(1) overstates by an order
    assert n_eff < 200.0


def test_effective_size_warns_when_the_sum_is_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A truncated sum returns an UPPER bound and must say so."""
    hours = np.arange(24 * 365, dtype=float)
    series = np.sin(2.0 * np.pi * hours / (24.0 * 400.0))  # never turns negative

    with caplog.at_level(logging.WARNING, logger="cooling_twin.analysis.residual"):
        effective_sample_size(series, max_lag=100)

    assert "UPPER BOUND" in caplog.text


def test_effective_size_never_exceeds_n_or_falls_below_one() -> None:
    """Correlation only ever costs information; it cannot create it."""
    rng = set_seed()
    series = rng.normal(0.0, 1.0, 5000)
    assert 1.0 <= effective_sample_size(series) <= series.size


def test_effective_size_rejects_an_impossible_max_lag() -> None:
    """A lag longer than the series would summarise nothing."""
    rng = set_seed()
    with pytest.raises(ValueError, match="shorter than the series"):
        effective_sample_size(rng.normal(0.0, 1.0, 100), max_lag=100)


def test_the_correction_inflates_band_standard_errors_exactly() -> None:
    """A quarter of the sample size doubles every standard error."""
    residual, temperature = _parabola(curvature=2.0)
    rng = set_seed()
    residual = residual + rng.normal(0.0, 100.0, residual.size)

    plain = fit_residual_curvature(residual, temperature, driver="d")
    corrected = fit_residual_curvature(
        residual, temperature, driver="d", effective_sample_ratio=0.25
    )

    assert corrected.effective_sample_ratio == 0.25
    for before, after in zip(plain.band_sems_kw, corrected.band_sems_kw, strict=True):
        assert after == pytest.approx(2.0 * before)
    # The means are untouched -- only the confidence in them changes.
    assert corrected.band_means_kw == pytest.approx(plain.band_means_kw)


def test_the_correction_can_overturn_a_u_verdict() -> None:
    """It must be capable of changing an answer, or it is decoration.

    A marginal U that clears its arms on the uncorrected standard errors
    must stop clearing them once the sample is known to be worth far
    less. If no ratio can flip the verdict, the correction is not doing
    anything and the test would be worthless.
    """
    rng = set_seed()
    n = 9000
    temperature = rng.uniform(0.0, 40.0, n)
    # A shallow parabola buried in heavy noise: real, but marginal.
    residual = 0.35 * (temperature - 20.0) ** 2 + rng.normal(0.0, 900.0, n)

    assert fit_residual_curvature(residual, temperature, driver="d").is_u_shaped
    assert not fit_residual_curvature(
        residual, temperature, driver="d", effective_sample_ratio=0.002
    ).is_u_shaped


@pytest.mark.parametrize("ratio", [0.0, -0.5, 1.5])
def test_an_impossible_effective_sample_ratio_raises(ratio: float) -> None:
    """n_eff / n cannot exceed 1 or reach 0."""
    residual, temperature = _parabola()
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        fit_residual_curvature(
            residual, temperature, driver="d", effective_sample_ratio=ratio
        )


# ---------------------------------------------------------------------
# matched-band split -- holding the confounder still
# ---------------------------------------------------------------------


def _confounded(
    n: int = 12000, probe_effect: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A probe that tracks the control, with a tunable effect of its own.

    Both the residual and the probe fall with temperature, so they
    correlate strongly whether or not the probe does anything. That is
    the trap the matched split exists to escape.
    """
    rng = set_seed()
    temperature = rng.uniform(-20.0, 35.0, n)
    probe = -3.0 * temperature + rng.normal(0.0, 5.0, n)
    residual = (
        -20.0 * temperature + probe_effect * probe + rng.normal(0.0, 30.0, n)
    )
    return residual, temperature, probe


def test_a_probe_with_no_effect_of_its_own_reads_flat() -> None:
    """The critical negative case.

    The probe here correlates with the residual at better than 0.9 and
    causes exactly none of it -- both are driven by temperature. A raw
    correlation would call this a finding; the matched split must not.
    """
    residual, temperature, probe = _confounded(probe_effect=0.0)

    assert np.corrcoef(residual, probe)[0, 1] > 0.9  # the trap

    split = matched_band_split(
        residual, temperature, probe, control="t", probe="p", control_width=2.5
    )

    assert not split.probe_raises_residual
    assert abs(split.weighted_difference_kw) < 10.0


def test_a_probe_with_a_real_effect_is_recovered() -> None:
    """Same confounding, but the probe now genuinely lifts the residual."""
    residual, temperature, probe = _confounded(probe_effect=2.0)

    split = matched_band_split(
        residual, temperature, probe, control="t", probe="p", control_width=2.5
    )

    assert split.probe_raises_residual
    # Within a bin the probe's spread is the +-5 noise, so the median
    # split separates halves about 8 units apart; at 2 kW per unit the
    # difference lands well clear of zero and of its own error.
    assert split.weighted_difference_kw > 4.0 * split.weighted_difference_sem_kw


def test_the_split_covers_the_control_range_it_was_given() -> None:
    """Bins are the control's, so their centres must span its range."""
    residual, temperature, probe = _confounded(probe_effect=1.0)

    split = matched_band_split(
        residual, temperature, probe, control="t", probe="p", control_width=2.5
    )

    assert split.centres.min() < -15.0
    assert split.centres.max() > 30.0
    assert np.all(split.counts_low >= MIN_BIN_COUNT)
    assert np.all(split.counts_high >= MIN_BIN_COUNT)
    assert split.differences == pytest.approx(split.means_high - split.means_low)


def test_the_split_is_invariant_to_the_probe_scale() -> None:
    """Units may be wrong (Q8), so only the probe's ORDER may matter."""
    residual, temperature, probe = _confounded(probe_effect=1.5)

    plain = matched_band_split(
        residual, temperature, probe, control="t", probe="p", control_width=2.5
    )
    rescaled = matched_band_split(
        residual,
        temperature,
        probe * 1000.0 + 7.0,
        control="t",
        probe="p",
        control_width=2.5,
    )

    assert rescaled.weighted_difference_kw == pytest.approx(
        plain.weighted_difference_kw
    )


def test_a_tied_probe_cannot_manufacture_a_difference() -> None:
    """A meter stuck at one value splits to nothing, not to noise."""
    rng = set_seed()
    n = 6000
    temperature = rng.uniform(0.0, 30.0, n)
    residual = rng.normal(0.0, 50.0, n)
    stuck = np.full(n, 42.0)

    with pytest.raises(ValueError, match="both sides of its"):
        matched_band_split(
            residual, temperature, stuck, control="t", probe="p", control_width=2.5
        )


def test_the_split_applies_the_effective_sample_correction() -> None:
    """A quarter of the sample doubles the standard error."""
    residual, temperature, probe = _confounded(probe_effect=1.0)

    plain = matched_band_split(
        residual, temperature, probe, control="t", probe="p", control_width=2.5
    )
    corrected = matched_band_split(
        residual,
        temperature,
        probe,
        control="t",
        probe="p",
        control_width=2.5,
        effective_sample_ratio=0.25,
    )

    assert corrected.weighted_difference_sem_kw == pytest.approx(
        2.0 * plain.weighted_difference_sem_kw
    )
    assert corrected.weighted_difference_kw == pytest.approx(
        plain.weighted_difference_kw
    )


def test_the_split_rejects_impossible_arguments() -> None:
    """Same validation vocabulary as the rest of the module."""
    residual, temperature, probe = _confounded()
    with pytest.raises(ValueError, match="control_width must be > 0"):
        matched_band_split(
            residual, temperature, probe, control="t", probe="p", control_width=0.0
        )
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        matched_band_split(
            residual,
            temperature,
            probe,
            control="t",
            probe="p",
            effective_sample_ratio=0.0,
        )
    with pytest.raises(ValueError, match="align element-wise"):
        matched_band_split(
            residual, temperature[:-1], probe, control="t", probe="p"
        )


def test_matched_split_to_frame_round_trips_the_numbers() -> None:
    """The table a report prints must be the split's own numbers."""
    residual, temperature, probe = _confounded(probe_effect=1.5)

    split = matched_band_split(
        residual, temperature, probe, control="t", probe="p", control_width=2.5
    )
    frame = split.to_frame()

    assert list(frame.index) == list(split.centres)
    assert frame["difference kW"].to_numpy() == pytest.approx(split.differences)
    assert int(frame[["hours low", "hours high"]].to_numpy().sum()) == int(
        split.counts_low.sum() + split.counts_high.sum()
    )


def test_matched_split_drops_bins_that_cannot_be_halved() -> None:
    """A bin with too few hours is skipped, not halved into noise."""
    rng = set_seed()
    n = 4000
    # A dense body plus a sparse tail: the tail bin holds fewer than
    # 2 * MIN_BIN_COUNT hours and must not reach the result.
    temperature = np.concatenate(
        [rng.uniform(0.0, 10.0, n), rng.uniform(40.0, 42.0, 25)]
    )
    probe = rng.normal(0.0, 1.0, temperature.size)
    residual = rng.normal(0.0, 10.0, temperature.size)

    split = matched_band_split(
        residual, temperature, probe, control="t", probe="p", control_width=2.5
    )

    assert split.centres.max() < 15.0
    assert np.all(split.counts_low >= MIN_BIN_COUNT)

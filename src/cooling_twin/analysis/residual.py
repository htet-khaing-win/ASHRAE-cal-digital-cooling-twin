"""Residual decomposition -- finding where the model's error lives (L7.1).

M6 produced a single number per building: CV(RMSE) 11.58% on the
held-out year. That number says how big the error is. It says nothing
about WHERE it is, and "where" is the only form of the answer that can
be acted on -- a model that is 11% wrong uniformly is finished, and a
model that is 3% wrong in winter and 25% wrong in July has a missing
term that happens to average out.

The residual is the whole of what the model failed to explain:

    residual = measured - predicted            [kW]

The sign convention is G14's, the same one `calibration/metrics.py`
uses, and it is kept here deliberately so that a residual and an NMBE
never disagree about direction:

    residual > 0  ->  the model UNDER-predicts (measured exceeds model)
    residual < 0  ->  the model OVER-predicts

If that residual is unstructured noise, the model has taken everything
the data contains. If it has structure -- a shape against time, load,
temperature, humidity or hour of day -- then the model is missing a
term, and the driver the structure lines up with names it. That is what
this module measures, and it is the mechanism behind Q7/Q8/ADR-011:
Claude's weather response is ~51.5 kW/K too flat, which was found
exactly this way.

WHAT THIS MODULE DOES NOT CLAIM. Every profile here is MARGINAL: the
mean residual within each bin of one driver, ignoring the others. Real
drivers are correlated (in Tempe the hot hours are also the occupied
hours), so a structure that belongs to one driver appears, weakened, in
every driver correlated with it. A marginal profile localises the error;
it does not attribute it. Attribution needs the residual's own
autocorrelation (L7.2) and a joint model of the drivers (L7.3).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)

# Bins holding fewer hours than this are dropped rather than reported.
# A bin mean over 4 hours is not a measurement of anything, but plotted
# next to bins of 800 hours it looks exactly as authoritative. 20 is the
# same floor scripts/diagnose_crossval_fold.py already uses for its
# binned-mean lines, kept identical so the two never disagree.
MIN_BIN_COUNT = 20

# A driver is called "structured" when its binned means explain at least
# this multiple of what pure noise would explain by chance (see
# `ResidualProfile.noise_floor`). 3.0 is a deliberate margin, not a
# significance test: binned means on 8,760 hours are easy to over-read,
# and this project's history is of thresholds set loosely and regretted.
MIN_STRUCTURE_RATIO = 3.0

# Fixed-width bin for outdoor dry bulb, K. Wide enough to hold hundreds
# of hours per bin on a full year, narrow enough that a cooling season
# spans ~10 bins.
TEMPERATURE_BIN_WIDTH_K = 2.5

# Equal-count bins for the driver whose distribution is building-
# specific. 10 gives deciles: enough resolution to see a curve, few
# enough that every bin is far above MIN_BIN_COUNT on a year of hours.
DEFAULT_QUANTILE_BINS = 10

# Below this many hours the binned means are not worth computing -- a
# week of data spread over 24 hour-of-day bins gives 7 points per bin.
MIN_HOURS = 24 * 14

_PERCENT = 100.0
_G_PER_KG = 1000.0


class Binning(Enum):
    """How a driver's values are grouped into bins.

    An enum rather than a string for the same reason `DataInterval` is
    one: a typo becomes an error at import time instead of a silently
    different -- and entirely plausible-looking -- profile.

    Attributes:
        CATEGORICAL: The values ARE the bins. For cyclic integer drivers
            (hour of day, month) where an ordering exists but distance
            does not: hour 23 and hour 0 are adjacent, and any binning
            that treats them as 23 units apart is wrong.
        FIXED_WIDTH: Equal-width bins in the driver's own units. Use
            when a bin must mean the same thing across buildings and
            years -- a "35.0-37.5 degC" bin is the same physical
            condition in Tempe and in Austin.
        QUANTILE: Equal-count bins. Use when the distribution is
            building-specific and fixed-width bins would leave the tails
            nearly empty and the middle overloaded.
    """

    CATEGORICAL = "categorical"
    FIXED_WIDTH = "fixed_width"
    QUANTILE = "quantile"


@dataclass(frozen=True, eq=False)
class ResidualProfile:
    """The mean residual within each bin of one driver.

    Frozen so a profile cannot be edited between computation and the
    report that quotes it; `eq=False` because the generated `__eq__`
    would compare NumPy arrays and raise the "truth value of an array is
    ambiguous" error -- the same choice `BuildingTimeSeries` made.

    Attributes:
        driver: Name of the driver, as it appears in reports.
        unit: Its unit, for axis labels and tables.
        centres: Mean value of the driver within each retained bin, in
            the driver's units. The MEAN of the data in the bin, not the
            nominal bin centre -- in a sparse tail the two differ, and
            the nominal one plots a point where no data sits.
        counts: Hours in each retained bin.
        means: Mean residual in each retained bin, kW.
        sems: Standard error of each bin mean, kW.
        explained_fraction: Share of residual variance the binned means
            account for (the correlation ratio, eta-squared), computed
            over the RETAINED hours only.
        noise_floor: What `explained_fraction` would be, in expectation,
            if the residual were pure noise -- `(k - 1) / (n - 1)` for
            `k` bins over `n` hours. Reported beside the value it judges
            because eta-squared is biased upward by bin count, so a
            bare 0.02 is meaningless until you know whether the floor is
            0.001 or 0.02.
        normaliser_kw: Mean measured load the swing is expressed against.
    """

    driver: str
    unit: str
    centres: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    means: npt.NDArray[np.float64]
    sems: npt.NDArray[np.float64]
    explained_fraction: float
    noise_floor: float
    normaliser_kw: float

    @property
    def swing_kw(self) -> float:
        """Largest bin mean minus smallest -- the size of the structure."""
        if self.means.size == 0:
            return 0.0
        return float(self.means.max() - self.means.min())

    @property
    def swing_pct_of_mean_load(self) -> float:
        """`swing_kw` as a percentage of mean measured load.

        The comparable form. A 500 kW swing is a rounding error on
        Claude (mean 6,884 kW) and a structural fault on a building a
        tenth the size, and a table mixing buildings must not invite
        that misreading.
        """
        return self.swing_kw / self.normaliser_kw * _PERCENT

    @property
    def structure_ratio(self) -> float:
        """`explained_fraction` in units of its own noise floor."""
        if self.noise_floor <= 0.0:
            return 0.0
        return self.explained_fraction / self.noise_floor

    @property
    def structured(self) -> bool:
        """Whether this driver carries structure worth acting on."""
        return self.structure_ratio >= MIN_STRUCTURE_RATIO

    def to_frame(self) -> pd.DataFrame:
        """The profile as a table, one row per retained bin."""
        return pd.DataFrame(
            {
                self.driver: self.centres,
                "hours": self.counts,
                "mean residual kW": self.means,
                "std error kW": self.sems,
            }
        ).set_index(self.driver)


@dataclass(frozen=True, eq=False)
class ResidualDecomposition:
    """One model's residual, profiled against every driver at once.

    Attributes:
        label: What was decomposed, e.g. `"Fox_education_Claude 2016"`.
        index: Timestamps of the residual, kept for L7.2's
            autocorrelation and for plotting the series against time.
        residual_kw: Measured minus predicted, kW, in index order.
        mean_measured_kw: Mean measured load over the same hours.
        profiles: One `ResidualProfile` per driver, keyed by driver name.
    """

    label: str
    index: pd.DatetimeIndex
    residual_kw: npt.NDArray[np.float64]
    mean_measured_kw: float
    profiles: Mapping[str, ResidualProfile]

    @property
    def mean_residual_kw(self) -> float:
        """Mean residual, kW. Equivalent to NMBE before normalisation."""
        return float(self.residual_kw.mean())

    @property
    def structured_drivers(self) -> tuple[str, ...]:
        """Drivers carrying structure, worst first."""
        ranked = sorted(
            (p for p in self.profiles.values() if p.structured),
            key=lambda p: p.structure_ratio,
            reverse=True,
        )
        return tuple(profile.driver for profile in ranked)

    def summary(self) -> pd.DataFrame:
        """One row per driver, most structured first.

        Returns:
            A table carrying, per driver: the share of residual variance
            its binned means explain, the noise floor that share is
            judged against, the ratio of the two, the peak-to-trough
            swing in kW and as a percentage of mean load, and the
            structured verdict.
        """
        rows = [
            {
                "driver": profile.driver,
                "unit": profile.unit,
                "bins": int(profile.counts.size),
                "explained": round(profile.explained_fraction, 4),
                "noise floor": round(profile.noise_floor, 4),
                "ratio": round(profile.structure_ratio, 1),
                "swing kW": round(profile.swing_kw, 0),
                "swing % load": round(profile.swing_pct_of_mean_load, 1),
                "structured": profile.structured,
            }
            for profile in self.profiles.values()
        ]
        frame = pd.DataFrame(rows).set_index("driver")
        return frame.sort_values("ratio", ascending=False)


def _validated_pair(
    residual_kw: npt.ArrayLike, driver_values: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate one residual/driver pair and return both as float arrays.

    Raises:
        ValueError: If the arrays differ in length, are empty, or hold
            non-finite values.
    """
    residual = np.asarray(residual_kw, dtype=float)
    driver = np.asarray(driver_values, dtype=float)
    if residual.shape != driver.shape:
        raise ValueError(
            f"residual has shape {residual.shape} and the driver "
            f"{driver.shape}; they must align element-wise"
        )
    if residual.size == 0:
        raise ValueError("cannot profile an empty residual")
    if not np.isfinite(residual).all() or not np.isfinite(driver).all():
        raise ValueError(
            "residual and driver must both be finite. NaN removal belongs "
            "in the M3 cleaning pipeline, not here -- dropping hours at "
            "this point would silently change which hours the profile "
            "covers and make it disagree with the CV(RMSE) it sits beside."
        )
    return residual, driver


def _bin_codes(
    driver: npt.NDArray[np.float64],
    binning: Binning,
    n_bins: int,
    width: float,
) -> npt.NDArray[np.int64]:
    """Assign each hour to a bin, returning integer codes.

    Raises:
        ValueError: If a quantile binning collapses to fewer than two
            distinct bins, or `width`/`n_bins` is not positive.
    """
    if binning is Binning.CATEGORICAL:
        rounded = np.rint(driver)
        if not np.allclose(rounded, driver):
            raise ValueError(
                "CATEGORICAL binning expects integer-valued driver data "
                "(hour of day, month). Use FIXED_WIDTH or QUANTILE for a "
                "continuous driver."
            )
        return rounded.astype(np.int64)

    if binning is Binning.FIXED_WIDTH:
        if width <= 0.0:
            raise ValueError(f"fixed-width binning needs width > 0, got {width}")
        return np.floor(driver / width).astype(np.int64)

    if n_bins < 2:
        raise ValueError(f"quantile binning needs n_bins >= 2, got {n_bins}")
    edges = np.unique(np.quantile(driver, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 3:
        # A driver that never moves explains nothing, which is a true
        # statement about it rather than an error. Raising here would
        # make the whole decomposition fail on a legitimate model -- the
        # L6.4 annual-mean baseline predicts one constant, and its
        # residual is exactly the kind of thing worth profiling. One bin
        # falls through to explained_fraction 0.0 on its own arithmetic.
        logger.warning(
            "quantile binning collapsed to a single bin: the driver is "
            "(nearly) constant, so it explains none of the residual by "
            "construction. Check this is the driver you meant to pass."
        )
        return np.zeros_like(driver, dtype=np.int64)
    # `side="right"` then clip: puts the maximum value in the last bin
    # rather than in a bin of its own, which would hold one point.
    return np.clip(
        np.searchsorted(edges, driver, side="right") - 1, 0, edges.size - 2
    ).astype(np.int64)


def residual_profile(
    residual_kw: npt.ArrayLike,
    driver_values: npt.ArrayLike,
    *,
    name: str,
    unit: str,
    binning: Binning,
    normaliser_kw: float,
    n_bins: int = DEFAULT_QUANTILE_BINS,
    width: float = TEMPERATURE_BIN_WIDTH_K,
    min_bin_count: int = MIN_BIN_COUNT,
) -> ResidualProfile:
    """Profile a residual against one driver.

    Args:
        residual_kw: Measured minus predicted, kW.
        driver_values: The driver, aligned element-wise with the
            residual. Never pass MEASURED load here -- see the warning
            below.
        name: Driver name for tables and axis labels.
        unit: Driver unit.
        binning: How to group the driver's values.
        normaliser_kw: Mean measured load, for `swing_pct_of_mean_load`.
        n_bins: Bin count, `QUANTILE` only.
        width: Bin width in the driver's units, `FIXED_WIDTH` only.
        min_bin_count: Bins holding fewer hours are dropped.

    Returns:
        The binned profile, with its explained variance share and the
        noise floor that share is judged against.

    Raises:
        ValueError: If the inputs do not align, are empty or non-finite,
            if `normaliser_kw` is not positive, if the binning arguments
            are invalid, or if no bin survives `min_bin_count`.

    Warning:
        Binning on MEASURED load manufactures structure out of nothing.
        Measurement noise sits on both sides -- it is inside the
        residual AND inside the bin assignment -- so hours that read
        high read high partly because their noise was positive, and
        their mean residual is positive for that reason alone. A
        perfect model scores a clean slope. Bin on PREDICTED load, which
        is independent of the measurement noise. `tests/test_residual.py`
        pins this with both versions of the same data.
    """
    residual, driver = _validated_pair(residual_kw, driver_values)
    if not np.isfinite(normaliser_kw) or normaliser_kw <= 0.0:
        raise ValueError(
            f"normaliser_kw must be a positive mean load, got {normaliser_kw}"
        )

    codes = _bin_codes(driver, binning, n_bins, width)
    unique, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    keep = counts >= min_bin_count
    if not keep.any():
        raise ValueError(
            f"no bin of {name} holds the required {min_bin_count} hours "
            f"(largest holds {int(counts.max())}). Widen the bins or pass a "
            "lower min_bin_count -- but a profile built on bins this thin is "
            "reporting sampling noise as structure."
        )
    if not keep.all():
        logger.info(
            "%s: dropped %d of %d bins holding fewer than %d hours",
            name,
            int((~keep).sum()),
            int(unique.size),
            min_bin_count,
        )

    kept_codes = unique[keep]
    retained = np.isin(codes, kept_codes)
    residual = residual[retained]
    driver = driver[retained]
    inverse = inverse[retained]
    # Re-index the surviving bins to 0..k-1 so `np.bincount` stays dense.
    remap = np.searchsorted(kept_codes, unique[inverse])

    n_kept = np.bincount(remap, minlength=kept_codes.size).astype(np.int64)
    sums = np.bincount(remap, weights=residual, minlength=kept_codes.size)
    means = sums / n_kept
    centres = np.bincount(remap, weights=driver, minlength=kept_codes.size) / n_kept

    deviations = residual - means[remap]
    within_sq = np.bincount(remap, weights=deviations**2, minlength=kept_codes.size)
    # ddof=1 within each bin; bins of exactly 1 hour cannot occur because
    # min_bin_count >= 1 is enforced by the caller and defaults to 20.
    variances = np.where(n_kept > 1, within_sq / np.maximum(n_kept - 1, 1), 0.0)
    sems = np.sqrt(variances / n_kept)

    grand_mean = residual.mean()
    total_ss = float(((residual - grand_mean) ** 2).sum())
    between_ss = float((n_kept * (means - grand_mean) ** 2).sum())
    # A residual with no variance at all (a perfect model, or a constant
    # bias) has nothing for any driver to explain. 0/0 is 0 here, not nan:
    # "this driver explains none of it" is the true statement.
    explained = 0.0 if total_ss == 0.0 else between_ss / total_ss

    k, n = int(kept_codes.size), int(residual.size)
    noise_floor = (k - 1) / (n - 1) if n > 1 else 0.0

    return ResidualProfile(
        driver=name,
        unit=unit,
        centres=centres,
        counts=n_kept,
        means=means,
        sems=sems,
        explained_fraction=float(explained),
        noise_floor=float(noise_floor),
        normaliser_kw=float(normaliser_kw),
    )


def decompose_residual(
    index: pd.DatetimeIndex,
    measured_kw: npt.ArrayLike,
    predicted_kw: npt.ArrayLike,
    *,
    t_ambient_c: npt.ArrayLike,
    humidity_ratio_kg_per_kg: npt.ArrayLike,
    label: str = "",
) -> ResidualDecomposition:
    """Decompose one model's error against the five standard drivers.

    The five are chosen so that each one, if it carries the structure,
    names a different missing term:

        | driver          | structure there means                      |
        |-----------------|--------------------------------------------|
        | month           | a seasonal term is missing or mis-sized     |
        | hour of day     | a schedule the model has no term for        |
        | predicted load  | the model's gain is wrong, not its offset   |
        | outdoor dry bulb| the weather response is too flat or steep   |
        | humidity ratio  | the latent/ventilation term is mis-sized    |

    Args:
        index: Hourly timestamps, aligned with the series.
        measured_kw: Metered cooling load, kW.
        predicted_kw: Model output over the same hours, kW.
        t_ambient_c: Outdoor dry bulb, degC.
        humidity_ratio_kg_per_kg: Outdoor humidity ratio, kg/kg. Passed
            in SI and reported in g/kg, because a slope of
            "113 kW per g/kg" is readable and one of "113,000 kW per
            kg/kg" is not.
        label: What is being decomposed, for reports.

    Returns:
        The decomposition, with one profile per driver.

    Raises:
        ValueError: If `index` is not a `DatetimeIndex`, if any series
            is misaligned, empty or non-finite, if fewer than
            `MIN_HOURS` hours are supplied, or if mean measured load is
            not positive.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            f"index must be a pandas DatetimeIndex, got {type(index).__name__}. "
            "Hour-of-day and month profiles are derived from it, and deriving "
            "them from positional assumptions is how an off-by-one in the "
            "timezone join becomes a fictitious 'schedule' in the residual."
        )
    measured = np.asarray(measured_kw, dtype=float)
    predicted = np.asarray(predicted_kw, dtype=float)
    if not (len(index) == measured.size == predicted.size):
        raise ValueError(
            f"index ({len(index)}), measured ({measured.size}) and predicted "
            f"({predicted.size}) must all be the same length"
        )
    if measured.size < MIN_HOURS:
        raise ValueError(
            f"{measured.size} hours is too few to profile; need at least "
            f"{MIN_HOURS}. Below that the hour-of-day bins hold single-digit "
            "counts and every profile is sampling noise."
        )
    if not np.isfinite(measured).all() or not np.isfinite(predicted).all():
        raise ValueError("measured and predicted must both be finite")

    mean_measured = float(measured.mean())
    if mean_measured <= 0.0:
        raise ValueError(
            f"mean measured load is {mean_measured:.1f} kW; the swing "
            "percentages are normalised by it and are undefined at or below "
            "zero"
        )

    residual = measured - predicted
    humidity = np.asarray(humidity_ratio_kg_per_kg, dtype=float) * _G_PER_KG

    profiles = {
        profile.driver: profile
        for profile in (
            residual_profile(
                residual,
                index.month.to_numpy(dtype=float),
                name="month",
                unit="1-12",
                binning=Binning.CATEGORICAL,
                normaliser_kw=mean_measured,
            ),
            residual_profile(
                residual,
                index.hour.to_numpy(dtype=float),
                name="hour_of_day",
                unit="0-23",
                binning=Binning.CATEGORICAL,
                normaliser_kw=mean_measured,
            ),
            residual_profile(
                residual,
                # PREDICTED, never measured -- see residual_profile's warning.
                predicted,
                name="predicted_load",
                unit="kW",
                binning=Binning.QUANTILE,
                normaliser_kw=mean_measured,
            ),
            residual_profile(
                residual,
                t_ambient_c,
                name="outdoor_dry_bulb",
                unit="degC",
                binning=Binning.FIXED_WIDTH,
                normaliser_kw=mean_measured,
                width=TEMPERATURE_BIN_WIDTH_K,
            ),
            residual_profile(
                residual,
                humidity,
                name="humidity_ratio",
                unit="g/kg",
                binning=Binning.QUANTILE,
                normaliser_kw=mean_measured,
            ),
        )
    }

    decomposition = ResidualDecomposition(
        label=label,
        index=index,
        residual_kw=residual,
        mean_measured_kw=mean_measured,
        profiles=MappingProxyType(profiles),
    )
    logger.info(
        "%s: mean residual %+.1f kW on %.1f kW mean load; structured drivers: %s",
        label or "residual",
        decomposition.mean_residual_kw,
        mean_measured,
        ", ".join(decomposition.structured_drivers) or "none",
    )
    return decomposition


def _demo() -> None:
    """Decompose a residual whose missing term is known by construction.

    A synthetic building is driven by weather AND by an occupancy
    schedule. The "model" is given the weather term exactly and the
    schedule not at all, so the residual IS the schedule plus noise --
    and the decomposition has to find it in the hour-of-day profile
    without being told.
    """
    from cooling_twin import set_seed

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    rng = set_seed()

    hours = 8760
    index = pd.date_range("2016-01-01", periods=hours, freq="h", tz="UTC")
    day_of_year = index.dayofyear.to_numpy(dtype=float)
    hour_of_day = index.hour.to_numpy(dtype=float)

    # Tempe-like: hot, dry, strong diurnal swing, seasonal humidity.
    t_ambient = (
        24.0
        + 12.0 * np.sin(2.0 * np.pi * (day_of_year - 100.0) / 365.0)
        + 6.0 * np.sin(2.0 * np.pi * (hour_of_day - 9.0) / 24.0)
    )
    humidity_ratio = 0.006 + 0.003 * np.sin(
        2.0 * np.pi * (day_of_year - 150.0) / 365.0
    )

    occupied = (
        (index.dayofweek.to_numpy() < 5) & (hour_of_day >= 7.0) & (hour_of_day < 19.0)
    ).astype(float)

    weather_term = 3000.0 + 180.0 * (t_ambient - 20.0)
    schedule_term = 900.0 * occupied
    measured = weather_term + schedule_term + rng.normal(0.0, 150.0, hours)
    predicted = weather_term  # the model has no schedule term at all

    decomposition = decompose_residual(
        index,
        measured,
        predicted,
        t_ambient_c=t_ambient,
        humidity_ratio_kg_per_kg=humidity_ratio,
        label="synthetic building, schedule term omitted",
    )

    print("\n--- 1. where the error lives ---")
    print(decomposition.summary().to_string())
    print(
        f"\nmean residual {decomposition.mean_residual_kw:+.1f} kW "
        f"on {decomposition.mean_measured_kw:.1f} kW mean load"
    )

    print("\n--- 2. the hour-of-day profile, which is the injected term ---")
    print(decomposition.profiles["hour_of_day"].to_frame().round(1).to_string())

    print("\n--- 3. the same residual against a driver the model HAS ---")
    print(decomposition.profiles["outdoor_dry_bulb"].to_frame().round(1).to_string())
    print(
        "\nOutdoor dry bulb shows structure too, and it is NOT a second missing\n"
        "term: occupied hours are also warm hours on this site, so the schedule\n"
        "signal leaks into every driver correlated with it. Marginal profiles\n"
        "localise an error; separating confounded drivers is L7.2 and L7.3."
    )


if __name__ == "__main__":
    _demo()

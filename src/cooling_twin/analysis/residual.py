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


def linear_residual_slopes(
    residual_kw: npt.ArrayLike,
    t_ambient_c: npt.ArrayLike,
    humidity_ratio_kg_per_kg: npt.ArrayLike,
) -> dict[str, float]:
    """Regress the residual on the two drivers the model already has.

    The binned profiles say WHICH driver carries the structure. This says
    HOW MUCH, in the units the missing term would be written in -- kW per
    K of outdoor temperature, kW per g/kg of humidity ratio. Those are
    the numbers that size a fix: a 51.5 kW/K shortfall can be compared
    directly against the model's own steady-state sensible slope, and a
    conclusion drawn about whether any admissible parameter set could
    close it. A ranking cannot be compared to anything.

    Both are reported from ONE multiple regression rather than two
    separate ones. Dry bulb and humidity ratio are correlated on every
    site in this project, and two univariate slopes would each carry the
    other's effect -- they would not sum to the joint response, and
    quoting them side by side would double-count.

    The intercept is reported beside the slopes, and the mean beside
    both, because a steep slope on a residual whose mean dominates is a
    second-order finding presented as the main one.

    Args:
        residual_kw: Measured minus predicted, kW.
        t_ambient_c: Outdoor dry bulb, degC.
        humidity_ratio_kg_per_kg: Outdoor humidity ratio, kg/kg. Reported
            per g/kg, so the slope is a readable number.

    Returns:
        Mean, intercept, both slopes, and the marginal correlation behind
        each slope.

    Raises:
        ValueError: If the arrays do not align, are empty or non-finite.
    """
    residual, temperature = _validated_pair(residual_kw, t_ambient_c)
    _, humidity = _validated_pair(residual_kw, humidity_ratio_kg_per_kg)

    # The design matrix uses g/kg so the slope is readable; the
    # correlation is computed on the ORIGINAL kg/kg. Correlation is
    # scale-invariant in exact arithmetic, but not to the last bit in
    # floating point, and this function replaced a script-local copy
    # whose recorded artifact must still reproduce exactly.
    design = np.column_stack(
        [np.ones_like(residual), temperature, humidity * _G_PER_KG]
    )
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    return {
        "mean_residual_kw": float(residual.mean()),
        "intercept_kw": float(coefficients[0]),
        "slope_kw_per_K": float(coefficients[1]),
        "slope_kw_per_g_per_kg": float(coefficients[2]),
        "corr_temperature": float(np.corrcoef(residual, temperature)[0, 1]),
        "corr_humidity": float(np.corrcoef(residual, humidity)[0, 1]),
    }


@dataclass(frozen=True, eq=False)
class CurvatureFit:
    """Whether a residual bends against a driver, and by how much.

    L7.1b found Claude's residual rising at BOTH ends of the outdoor
    temperature range and falling in the middle. A linear slope cannot
    express that -- the two arms cancel and it reports the average of a
    shape it has no way to represent. Two independent statements are
    made here instead, one parametric and one not, because either alone
    is arguable:

      * the quadratic term, which says the shape bends and which way;
      * the three band means, which make no functional assumption at all
        and which a reviewer can check against the binned profile by eye.

    A U is only claimed when BOTH ends exceed the middle by more than the
    combined standard error. A quadratic term alone is not enough: fitting
    a parabola to a straight line with one outlying bin also returns a
    positive coefficient.

    Attributes:
        driver: Driver the residual was fitted against.
        quadratic_kw_per_unit2: Second-order coefficient. Positive is
            convex (U), negative is concave (inverted U).
        linear_kw_per_unit: First-order coefficient, at the same fit.
        intercept_kw: Constant term.
        vertex: Driver value at the turning point, or None if the fit is
            effectively straight.
        r_squared: Share of residual variance the quadratic explains.
        linear_r_squared: The same for a straight line, so the gain from
            the curvature term is visible rather than asserted.
        band_edges: `(lower, upper)` driver values splitting the three
            bands.
        band_means_kw: Mean residual in the low, middle and high bands.
        band_sems_kw: Standard error of each band mean, ALREADY inflated
            by `effective_sample_ratio`.
        band_counts: Hours in each band.
        effective_sample_ratio: `n_eff / n` used to inflate the standard
            errors. 1.0 means no correction was applied -- correct for
            independent data and WRONG for an hourly residual, which is
            why it is recorded rather than assumed.
    """

    driver: str
    quadratic_kw_per_unit2: float
    linear_kw_per_unit: float
    intercept_kw: float
    vertex: float | None
    r_squared: float
    linear_r_squared: float
    band_edges: tuple[float, float]
    band_means_kw: tuple[float, float, float]
    band_sems_kw: tuple[float, float, float]
    band_counts: tuple[int, int, int]
    effective_sample_ratio: float = 1.0

    @property
    def low_lift_kw(self) -> float:
        """How far the low band sits above the middle one."""
        return self.band_means_kw[0] - self.band_means_kw[1]

    @property
    def high_lift_kw(self) -> float:
        """How far the high band sits above the middle one."""
        return self.band_means_kw[2] - self.band_means_kw[1]

    @property
    def is_u_shaped(self) -> bool:
        """Both ends above the middle, by more than their combined error."""
        low_error = self.band_sems_kw[0] + self.band_sems_kw[1]
        high_error = self.band_sems_kw[2] + self.band_sems_kw[1]
        return (
            self.low_lift_kw > U_SHAPE_SIGMA * low_error
            and self.high_lift_kw > U_SHAPE_SIGMA * high_error
            and self.quadratic_kw_per_unit2 > 0.0
        )


# How many combined standard errors each arm of a U must clear before the
# shape is called real. 2.0 is a deliberate margin over the ~1.96 of a
# conventional 95% interval, and it is applied to the SUM of two standard
# errors rather than to a pooled one -- the conservative reading of a
# difference between two means.
U_SHAPE_SIGMA = 2.0

# Fraction of hours in each of the outer bands. Terciles: equal hours in
# cold, shoulder and hot, computed on the building's OWN distribution.
#
# The obvious alternative is fixed physical thresholds ("below 10 degC is
# cold"). Rejected: 03_DOMAIN_REFERENCE.md sets no balance-point
# temperature, so any such number would be invented, and it would not
# survive contact with this portfolio anyway -- Cathleen runs from -24 to
# +33 degC and Claude from +4 to +44, so one fixed edge puts most of one
# building in a band that is empty for the other. Terciles are a
# statement about each building's own operating range, apply identically
# to every building, and flatter none of them. Same discipline as
# `calibration/bounds.py`'s derived internal-gain bound.
BAND_QUANTILE = 1.0 / 3.0


def band_edges_from_quantiles(
    driver_values: npt.ArrayLike, quantile: float = BAND_QUANTILE
) -> tuple[float, float]:
    """Tercile edges of a driver, for splitting low/middle/high bands.

    Computed ONCE, on the training year, and passed to every later call.
    Recomputing them per year would move the bands with the weather, and
    a train-versus-test comparison across bands that are not the same
    bands answers no question at all.

    Args:
        driver_values: The driver, e.g. outdoor dry bulb.
        quantile: Fraction of hours in each outer band.

    Returns:
        `(lower_edge, upper_edge)`.

    Raises:
        ValueError: If the driver is empty, non-finite, if `quantile` is
            not in `(0, 0.5)`, or if the two edges coincide.
    """
    driver = np.asarray(driver_values, dtype=float)
    if driver.size == 0:
        raise ValueError("cannot derive bands from an empty driver")
    if not np.isfinite(driver).all():
        raise ValueError("driver must be finite")
    if not 0.0 < quantile < 0.5:
        raise ValueError(f"quantile must lie in (0, 0.5), got {quantile}")
    lower, upper = np.quantile(driver, [quantile, 1.0 - quantile])
    if lower == upper:
        raise ValueError(
            "the two band edges coincide -- this driver does not vary "
            "enough to split into bands"
        )
    return float(lower), float(upper)


def fit_residual_curvature(
    residual_kw: npt.ArrayLike,
    driver_values: npt.ArrayLike,
    *,
    driver: str,
    band_edges: tuple[float, float] | None = None,
    effective_sample_ratio: float = 1.0,
) -> CurvatureFit:
    """Test a residual for curvature against one driver.

    Args:
        residual_kw: Measured minus predicted, kW.
        driver_values: The driver, aligned element-wise.
        driver: Driver name, for the report.
        band_edges: Band edges to use. Pass the TRAINING year's edges
            when fitting a later year, so both years are scored on the
            same bands. Derived from this data when omitted.
        effective_sample_ratio: `n_eff / n` from
            `effective_sample_size()`, which inflates the band standard
            errors by `sqrt(1 / ratio)`. Defaults to 1.0 -- no
            correction -- because this function cannot know whether it
            was handed a time series or a scatter of independent draws,
            and silently assuming the former would break the only cases
            where the right answer is known by construction. For an
            hourly residual, PASS IT: on this project's buildings the
            uncorrected standard errors are too small by 7 to 13 times.

    Returns:
        The quadratic fit and the three band means.

    Raises:
        ValueError: If the inputs do not align, are empty or non-finite,
            if a band holds fewer than `MIN_BIN_COUNT` hours, or if
            `effective_sample_ratio` is outside `(0, 1]`.
    """
    residual, values = _validated_pair(residual_kw, driver_values)
    if not 0.0 < effective_sample_ratio <= 1.0:
        raise ValueError(
            f"effective_sample_ratio must lie in (0, 1], got "
            f"{effective_sample_ratio}. It is n_eff / n, so it cannot "
            "exceed 1 -- correlation only ever costs information."
        )
    lower, upper = (
        band_edges_from_quantiles(values) if band_edges is None else band_edges
    )

    # Centred on the mean so the quadratic and linear terms are not
    # near-collinear. On a driver running 4 to 44 degC the raw design
    # matrix has a condition number in the thousands, and the reported
    # coefficients then depend on the solver's tolerance rather than on
    # the data. Centring does not change the fitted CURVE, only the basis
    # it is expressed in -- and the quadratic coefficient, which is what
    # is being reported, is identical either way.
    centre = float(values.mean())
    shifted = values - centre
    design = np.column_stack([np.ones_like(shifted), shifted, shifted**2])
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    quadratic = float(coefficients[2])

    total_ss = float(((residual - residual.mean()) ** 2).sum())
    quadratic_ss = float(((residual - design @ coefficients) ** 2).sum())
    linear_coefficients, *_ = np.linalg.lstsq(design[:, :2], residual, rcond=None)
    linear_ss = float(((residual - design[:, :2] @ linear_coefficients) ** 2).sum())
    r_squared = 0.0 if total_ss == 0.0 else 1.0 - quadratic_ss / total_ss
    linear_r_squared = 0.0 if total_ss == 0.0 else 1.0 - linear_ss / total_ss

    # Expressed back in the driver's own units, so the turning point can
    # be compared with a physical temperature -- the calibrated setpoint,
    # for instance.
    vertex = None if quadratic == 0.0 else centre - coefficients[1] / (2.0 * quadratic)

    masks = (values < lower, (values >= lower) & (values <= upper), values > upper)
    counts = tuple(int(mask.sum()) for mask in masks)
    if min(counts) < MIN_BIN_COUNT:
        raise ValueError(
            f"a {driver} band holds only {min(counts)} hours (need "
            f"{MIN_BIN_COUNT}). Band edges {lower:.2f}/{upper:.2f} do not "
            "suit this data -- most likely they came from a year whose "
            "range does not overlap this one."
        )
    means = tuple(float(residual[mask].mean()) for mask in masks)
    inflation = float(np.sqrt(1.0 / effective_sample_ratio))
    sems = tuple(
        float(residual[mask].std(ddof=1) / np.sqrt(mask.sum()) * inflation)
        for mask in masks
    )

    return CurvatureFit(
        driver=driver,
        quadratic_kw_per_unit2=quadratic,
        linear_kw_per_unit=float(coefficients[1]),
        intercept_kw=float(coefficients[0]),
        vertex=None if vertex is None else float(vertex),
        r_squared=r_squared,
        linear_r_squared=linear_r_squared,
        band_edges=(lower, upper),
        band_means_kw=(means[0], means[1], means[2]),
        band_sems_kw=(sems[0], sems[1], sems[2]),
        band_counts=(counts[0], counts[1], counts[2]),
        effective_sample_ratio=effective_sample_ratio,
    )


# Lags reported by default, in hours: the previous hour, the same hour
# yesterday, the same hour last week. Each names a different mechanism --
# thermal mass, a daily cycle the model does not reproduce, a weekly
# schedule -- so the three together say more than a single number.
DEFAULT_ACF_LAGS = (1, 24, 168)

# Lags entering the Ljung-Box statistic. A full week of hourly lags: long
# enough to catch a weekly cycle, and the conventional choice for hourly
# data with a daily structure.
DEFAULT_LJUNG_BOX_LAGS = 168

# Hours per day, for the daily-averaging test. A residual that is pure
# white noise loses variance in proportion to the averaging window, so
# 1/24 is the null this test is read against.
HOURS_PER_DAY = 24


@dataclass(frozen=True, eq=False)
class ResidualDiagnostics:
    """Is the residual random, or is there a model still inside it? (L7.2)

    Attributes:
        label: What was tested.
        n_hours: Hours in the series.
        acf: Autocorrelation at each reported lag, keyed by lag in hours.
        ljung_box_q: The Ljung-Box statistic over `ljung_box_lags`.
        ljung_box_lags: Number of lags in the statistic, its chi-square
            degrees of freedom.
        ljung_box_p: The p-value. Read the WARNING below before quoting
            it.
        daily_variance_share: Variance of the daily-mean residual divided
            by variance of the hourly residual.
        white_noise_variance_share: What that would be for white noise,
            `1 / 24`. The null the share above is judged against, for the
            same reason `ResidualProfile` carries a noise floor.

    Warning:
        The p-value is not the finding and must not be reported as one.
        At n = 8,760 the Ljung-Box test rejects white noise for
        autocorrelations far too small to matter, so p is effectively
        zero for every building energy residual ever measured. It is
        computed here because its ABSENCE would be conspicuous, and
        because a p that is NOT tiny would be genuinely surprising and
        worth investigating. The effect sizes -- the ACF values and the
        daily variance share -- are what carry the information.
    """

    label: str
    n_hours: int
    acf: Mapping[int, float]
    ljung_box_q: float
    ljung_box_lags: int
    ljung_box_p: float
    daily_variance_share: float
    white_noise_variance_share: float

    @property
    def survives_daily_averaging(self) -> bool:
        """Whether the error is systematic rather than measurement noise.

        Averaging 24 independent draws cuts variance 24-fold. An error
        that survives that is not noise -- it is a signal the model has
        not taken, and it is therefore learnable.
        """
        return self.daily_variance_share > MIN_STRUCTURE_RATIO * (
            self.white_noise_variance_share
        )


def autocorrelation(
    residual_kw: npt.ArrayLike, lags: tuple[int, ...] = DEFAULT_ACF_LAGS
) -> dict[int, float]:
    """Autocorrelation of a residual at the given lags.

    Uses the BIASED estimator -- the sum of lagged products divided by
    `n` rather than by `n - k`. That is the definition the Ljung-Box
    statistic is built on, and mixing the two would make the reported
    correlations disagree with the test computed from them. At the lags
    and sample sizes here the difference is a fraction of a percent; the
    consistency matters more than the fraction.

    Args:
        residual_kw: The residual, in index order. Must be evenly spaced
            in time -- gaps that were dropped rather than filled will
            silently shorten a lag.
        lags: Lags in samples.

    Returns:
        `{lag: correlation}`.

    Raises:
        ValueError: If the series is non-finite, or a lag is not a
            positive integer shorter than the series.
    """
    residual = np.asarray(residual_kw, dtype=float)
    if residual.ndim != 1 or residual.size == 0:
        raise ValueError("residual must be a non-empty one-dimensional series")
    if not np.isfinite(residual).all():
        raise ValueError("residual must be finite")

    centred = residual - residual.mean()
    denominator = float((centred**2).sum())
    if denominator == 0.0:
        raise ValueError(
            "residual has zero variance; autocorrelation is undefined. A "
            "perfectly constant residual is a bias, and L7.1's profiles "
            "are the tool for it."
        )

    result = {}
    for lag in lags:
        if lag < 1 or lag >= residual.size:
            raise ValueError(
                f"lag {lag} must be at least 1 and shorter than the "
                f"series ({residual.size} samples)"
            )
        result[lag] = float((centred[lag:] * centred[:-lag]).sum() / denominator)
    return result


@dataclass(frozen=True, eq=False)
class MatchedSplit:
    """Does a probe raise the residual once a confounder is held fixed?

    The question "is the winter residual caused by reheat?" cannot be
    answered by correlating the residual with a heating meter: both rise
    as it gets colder, so they correlate whether or not one causes the
    other. The confounder has to be held still first.

    So: bin on the CONTROL (outdoor temperature), and inside each bin
    split the hours at the median of the PROBE (the heating meter). Hours
    in the same bin are at nearly the same outdoor temperature, so a
    difference between the two halves is a difference the control cannot
    explain. This is the design L6.7b used to keep the humidity
    hypothesis alive and kill the schedule one; it is written down here
    so the next use is the same test rather than a similar one.

    The probe enters ONLY through a median split, so nothing depends on
    its units or its scale. That matters on this project: Q8 established
    that the Fox hot-water meter's magnitudes are wrong, and the same
    caution applies to any companion meter until proven otherwise. A
    rank-based split stays valid under any monotone unit error.

    Within each bin the control's own linear trend is removed from the
    residual before the split, because binning narrows a confounder
    without holding it still -- see the comment in `matched_band_split`.

    Attributes:
        control: Name of the confounder held fixed.
        probe: Name of the series being tested.
        centres: Mean control value in each retained bin.
        counts_low: Hours in the low-probe half of each bin.
        counts_high: Hours in the high-probe half of each bin.
        means_low: Mean residual of the low-probe half, kW, after the
            within-bin control trend is removed.
        means_high: The same for the high-probe half.
        differences: `means_high - means_low`, kW.
        weighted_difference_kw: Hours-weighted mean of `differences`.
        weighted_difference_sem_kw: Its standard error, already inflated
            by `effective_sample_ratio`.
        effective_sample_ratio: `n_eff / n` applied. See
            `effective_sample_size`.
    """

    control: str
    probe: str
    centres: npt.NDArray[np.float64]
    counts_low: npt.NDArray[np.int64]
    counts_high: npt.NDArray[np.int64]
    means_low: npt.NDArray[np.float64]
    means_high: npt.NDArray[np.float64]
    differences: npt.NDArray[np.float64]
    weighted_difference_kw: float
    weighted_difference_sem_kw: float
    effective_sample_ratio: float

    @property
    def probe_raises_residual(self) -> bool:
        """Whether the high-probe half sits above the low-probe half.

        Two standard errors, on the SAME margin `CurvatureFit` uses, so
        the two verdicts in this module cannot mean different things by
        the word "significant".
        """
        return self.weighted_difference_kw > U_SHAPE_SIGMA * (
            self.weighted_difference_sem_kw
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per control bin, for the report."""
        return pd.DataFrame(
            {
                self.control: self.centres,
                "hours low": self.counts_low,
                "hours high": self.counts_high,
                f"residual, low {self.probe}": self.means_low,
                f"residual, high {self.probe}": self.means_high,
                "difference kW": self.differences,
            }
        ).set_index(self.control)


def matched_band_split(
    residual_kw: npt.ArrayLike,
    control_values: npt.ArrayLike,
    probe_values: npt.ArrayLike,
    *,
    control: str,
    probe: str,
    control_width: float = TEMPERATURE_BIN_WIDTH_K,
    min_bin_count: int = MIN_BIN_COUNT,
    effective_sample_ratio: float = 1.0,
) -> MatchedSplit:
    """Split the residual by a probe, at matched values of a control.

    Args:
        residual_kw: Measured minus predicted, kW.
        control_values: The confounder to hold fixed, e.g. outdoor dry
            bulb. Binned at `control_width`.
        probe_values: The series under test, e.g. a heating meter. Used
            only for its ORDER within each bin -- units are irrelevant
            and may be wrong.
        control: Control name, for the report.
        probe: Probe name, for the report.
        control_width: Bin width in the control's units. Narrow enough
            that hours in a bin really are comparable, wide enough that
            each half clears `min_bin_count`.
        min_bin_count: Minimum hours required in EACH half of a bin for
            that bin to be retained.
        effective_sample_ratio: `n_eff / n`, inflating the standard
            error. See `fit_residual_curvature` for why this is an
            explicit argument rather than a silent correction.

    Returns:
        The per-bin split and its hours-weighted summary.

    Raises:
        ValueError: If the inputs do not align, are empty or non-finite,
            if `effective_sample_ratio` is outside `(0, 1]`, or if no
            bin has enough hours on both sides of its median.
    """
    residual, control_array = _validated_pair(residual_kw, control_values)
    _, probe_array = _validated_pair(residual_kw, probe_values)
    if not 0.0 < effective_sample_ratio <= 1.0:
        raise ValueError(
            f"effective_sample_ratio must lie in (0, 1], got {effective_sample_ratio}"
        )
    if control_width <= 0.0:
        raise ValueError(f"control_width must be > 0, got {control_width}")

    codes = np.floor(control_array / control_width).astype(np.int64)
    rows = []
    for code in np.unique(codes):
        in_bin = codes == code
        if in_bin.sum() < 2 * min_bin_count:
            continue
        probe_in_bin = probe_array[in_bin]
        # Strictly above the median on one side, at-or-below on the
        # other. A tied probe (a meter reading the same value for many
        # hours) then lands entirely in the low half rather than being
        # split arbitrarily, which keeps the comparison honest instead
        # of manufacturing a difference out of tie-breaking order.
        threshold = float(np.median(probe_in_bin))
        high = probe_in_bin > threshold
        low = ~high
        if high.sum() < min_bin_count or low.sum() < min_bin_count:
            continue

        # Remove the control's own linear trend WITHIN the bin before
        # splitting. Binning alone does not hold the control still, it
        # only narrows it: inside a 2.5 K bin the temperature still moves
        # 2.5 K, and a probe that tracks temperature therefore still
        # sorts hours by temperature after the split. The leakage scales
        # with bin width times the control's slope, so it is largest
        # exactly where the confounding is strongest -- which is where
        # this test is being relied on. A within-bin detrend removes the
        # linear part of it exactly. Verified by
        # test_a_probe_with_no_effect_of_its_own_reads_flat, which FAILS
        # without these four lines.
        control_in_bin = control_array[in_bin]
        design = np.column_stack([np.ones_like(control_in_bin), control_in_bin])
        coefficients, *_ = np.linalg.lstsq(design, residual[in_bin], rcond=None)
        residual_in_bin = residual[in_bin] - design @ coefficients

        mean_low = float(residual_in_bin[low].mean())
        mean_high = float(residual_in_bin[high].mean())
        variance = (
            residual_in_bin[low].var(ddof=1) / low.sum()
            + residual_in_bin[high].var(ddof=1) / high.sum()
        )
        rows.append(
            (
                float(control_array[in_bin].mean()),
                int(low.sum()),
                int(high.sum()),
                mean_low,
                mean_high,
                mean_high - mean_low,
                float(variance),
            )
        )

    if not rows:
        raise ValueError(
            f"no {control} bin of width {control_width} holds {min_bin_count} "
            f"hours on both sides of its {probe} median. Widen the bins, or "
            "accept that this comparison cannot be made on this data."
        )

    centres, low_counts, high_counts, low_means, high_means, differences, variances = (
        np.array(column) for column in zip(*rows, strict=True)
    )
    weights = low_counts + high_counts
    weighted = float((weights * differences).sum() / weights.sum())
    inflation = float(np.sqrt(1.0 / effective_sample_ratio))
    sem = (
        float(np.sqrt((weights**2 * variances).sum()) / weights.sum()) * inflation
    )

    split = MatchedSplit(
        control=control,
        probe=probe,
        centres=centres,
        counts_low=low_counts.astype(np.int64),
        counts_high=high_counts.astype(np.int64),
        means_low=low_means,
        means_high=high_means,
        differences=differences,
        weighted_difference_kw=weighted,
        weighted_difference_sem_kw=sem,
        effective_sample_ratio=effective_sample_ratio,
    )
    logger.info(
        "%s at matched %s: high half runs %+.1f kW (+/- %.1f) against the low "
        "half across %d bins -- %s",
        probe,
        control,
        weighted,
        sem,
        centres.size,
        "RAISES the residual" if split.probe_raises_residual else "no clear effect",
    )
    return split


def effective_sample_size(
    residual_kw: npt.ArrayLike, *, max_lag: int = DEFAULT_LJUNG_BOX_LAGS
) -> float:
    """How many INDEPENDENT observations a correlated series is worth.

    8,760 hourly residuals are not 8,760 measurements. When this hour's
    error is 0.83 correlated with the last one, most of those rows are
    repeating information already present, and any standard error
    computed as `s / sqrt(n)` is too small -- by a factor of three on
    Claude and nearly eight on Cathleen. Every band mean, confidence
    interval and significance claim made on a residual is wrong by that
    factor unless it is corrected.

    The estimator is the variance-inflation factor for a MEAN, which is
    the quantity actually needed here:

        n_eff = n / (1 + 2 * sum_k rho_k)

    summed to the first non-positive autocorrelation (the initial
    positive sequence), capped at `max_lag`. The simpler AR(1) form
    `n * (1 - rho_1) / (1 + rho_1)` is NOT used, because these residuals
    are nothing like AR(1): Claude's rho(1) of 0.826 would imply
    rho(24) = 0.826^24 = 0.010, and the measured value is 0.598. A
    long-memory series scored by an AR(1) correction gets an
    optimistically LARGE effective sample size, which is the direction
    that matters -- it would leave the standard errors still too small
    while appearing to have addressed the problem.

    Args:
        residual_kw: The residual, in index order, evenly spaced.
        max_lag: Where the sum is truncated if the autocorrelation has
            not gone non-positive by then.

    Returns:
        The effective sample size, at least 1.0 and at most `n`.

    Raises:
        ValueError: If the series is empty or non-finite, or `max_lag`
            is not a positive integer shorter than it.
    """
    residual = np.asarray(residual_kw, dtype=float)
    n = int(residual.size)
    if max_lag < 1 or max_lag >= n:
        raise ValueError(
            f"max_lag must be at least 1 and shorter than the series "
            f"({n} samples), got {max_lag}"
        )

    acf = autocorrelation(residual, tuple(range(1, max_lag + 1)))
    total = 0.0
    truncated_early = False
    for lag in range(1, max_lag + 1):
        if acf[lag] <= 0.0:
            truncated_early = True
            break
        total += acf[lag]

    if not truncated_early:
        # The sum was cut off while the correlations were still positive,
        # so the true inflation is LARGER than this and the returned
        # figure is an upper bound on the effective sample size. Said
        # plainly because the failure mode is silent: a number that is
        # optimistic in exactly the direction the caller is trying to
        # guard against.
        logger.warning(
            "autocorrelation was still positive (%.3f) at the %d-lag "
            "truncation, so the effective sample size returned is an "
            "UPPER BOUND -- the real one is smaller and every standard "
            "error derived from it is still optimistic.",
            acf[max_lag],
            max_lag,
        )

    inflation = 1.0 + 2.0 * total
    return float(min(max(n / inflation, 1.0), n))


def residual_diagnostics(
    residual_kw: npt.ArrayLike,
    *,
    label: str = "",
    lags: tuple[int, ...] = DEFAULT_ACF_LAGS,
    ljung_box_lags: int = DEFAULT_LJUNG_BOX_LAGS,
) -> ResidualDiagnostics:
    """Test whether a residual is white noise (L7.2).

    Three measurements, deliberately not one:

      1. Autocorrelation at named lags -- how much of this hour's error
         is predictable from an earlier one, and at which spacing.
      2. Ljung-Box over a week of lags -- the formal test, kept for
         completeness and read with the warning on
         `ResidualDiagnostics`.
      3. The share of variance surviving daily averaging, against the
         1/24 a white-noise residual would give. This is the one that
         decides whether the remaining error is worth modelling: noise
         averages away, structure does not.

    Args:
        residual_kw: The residual, in index order, evenly spaced.
        label: What is being tested, for the log.
        lags: Lags for the reported autocorrelations.
        ljung_box_lags: Lags entering the Ljung-Box statistic.

    Returns:
        The diagnostics.

    Raises:
        ValueError: If the series is empty or non-finite, or if
            `ljung_box_lags` is not a positive integer shorter than it.
    """
    from scipy.stats import chi2

    residual = np.asarray(residual_kw, dtype=float)
    n = int(residual.size)
    if ljung_box_lags < 1 or ljung_box_lags >= n:
        raise ValueError(
            f"ljung_box_lags must be at least 1 and shorter than the series "
            f"({n} samples), got {ljung_box_lags}"
        )

    reported = autocorrelation(residual, lags)
    every_lag = autocorrelation(residual, tuple(range(1, ljung_box_lags + 1)))

    # Q = n(n+2) * sum_k rho_k^2 / (n - k).
    # Degrees of freedom are the lag count, NOT lags minus fitted
    # parameters: the usual correction applies when the residual comes
    # from an ARMA fit ON THIS SERIES. Here it comes from a physical
    # model fitted to the load, which has consumed no autocorrelation
    # structure, so subtracting its 5 parameters would be borrowed
    # arithmetic from a different situation.
    statistic = float(
        n * (n + 2) * sum(rho**2 / (n - lag) for lag, rho in every_lag.items())
    )
    p_value = float(chi2.sf(statistic, ljung_box_lags))

    # Trailing partial day dropped rather than averaged over fewer hours,
    # which would give that one point a larger variance and inflate the
    # share this test reports.
    whole_days = n // HOURS_PER_DAY
    daily = residual[: whole_days * HOURS_PER_DAY].reshape(whole_days, HOURS_PER_DAY)
    daily_share = float(daily.mean(axis=1).var() / residual.var())

    diagnostics = ResidualDiagnostics(
        label=label,
        n_hours=n,
        acf=MappingProxyType(dict(reported)),
        ljung_box_q=statistic,
        ljung_box_lags=ljung_box_lags,
        ljung_box_p=p_value,
        daily_variance_share=daily_share,
        white_noise_variance_share=1.0 / HOURS_PER_DAY,
    )
    logger.info(
        "%s: ACF %s; Ljung-Box Q=%.0f (%d lags, p=%.3g); daily variance "
        "share %.3f against a white-noise %.3f",
        label or "residual",
        {lag: round(value, 3) for lag, value in reported.items()},
        statistic,
        ljung_box_lags,
        p_value,
        daily_share,
        1.0 / HOURS_PER_DAY,
    )
    return diagnostics


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

    print("\n--- 4. L7.2: is what is left random? ---")
    structured = residual_diagnostics(
        decomposition.residual_kw, label="residual WITH the schedule left in"
    )
    # The same model with the schedule term restored: the residual is
    # then the measurement noise alone, which is what "done" looks like.
    finished = residual_diagnostics(
        measured - (weather_term + schedule_term), label="residual with NOTHING left"
    )
    for name, diagnostics in (
        ("schedule left in", structured),
        ("nothing left    ", finished),
    ):
        print(
            f"  {name}  rho(1) {diagnostics.acf[1]:+.3f}  "
            f"rho(24) {diagnostics.acf[24]:+.3f}  "
            f"daily var share {diagnostics.daily_variance_share:.3f} "
            f"(white noise {diagnostics.white_noise_variance_share:.3f})  "
            f"LB p {diagnostics.ljung_box_p:.3f}  "
            f"survives: {diagnostics.survives_daily_averaging}"
        )
    print(
        f"\n  effective sample size, schedule left in: "
        f"{effective_sample_size(decomposition.residual_kw):.0f} of {hours}"
    )
    print(
        "\nLjung-Box does its job here: ~0 for the structured residual, ~0.98 for\n"
        "the finished one. What it CANNOT do is say how much structure there is.\n"
        "On this project's real buildings it returns 0 for all six building-years,\n"
        "including the one whose residual is a third the size of the others. It is\n"
        "a yes/no that saturates immediately, so the daily variance share and the\n"
        "autocorrelations are what carry the information."
    )


if __name__ == "__main__":
    _demo()

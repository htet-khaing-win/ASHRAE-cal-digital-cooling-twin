"""Deriving parameter bounds from the data instead of setting them by hand.

A bound is a physical statement (L6.5), and the way this project kept
getting them wrong was by anchoring them on the wrong measurement.

The internal-gain bound was first set from an office-like guess
(5-15 W/m2, L6.1), then widened to 200 on Q6's laboratory evidence, then
narrowed to 60 and 120 on the building's own electricity meter -- the
reasoning being that equipment and lighting gains cannot exceed the
electrical power that produces them. That reasoning has a hole:

    A load served DIRECTLY by chilled water -- a water-cooled condenser,
    process equipment, a machine room -- never appears on the building's
    electricity meter at all.

The hole is not a corner case. Measured on the three selected buildings,
the cooling load that remains when the weather contributes nothing is
2.2 to 3.5 times each building's entire electricity intensity:

    building                  mean    cold floor   electricity   ratio
    Fox_education_Claude    383 W/m2   166 W/m2     53.6 W/m2     3.1x
    Bull_education_Luke     171 W/m2    69 W/m2     19.5 W/m2     3.5x
    Hog_education_Cathleen   78 W/m2    33 W/m2     15.1 W/m2     2.2x

So this module replaces the anchor. Envelope conduction and ventilation
load both fall to zero as outdoor temperature approaches the setpoint,
which means whatever load survives the coldest hours is the constant
term, measured directly, with no model in the loop. That measurement is
what bounds the parameter.

The rule is deliberately mechanical: no per-building judgement, no hand
tuning, and it derives the SAME way for every building including the
ones it does not flatter.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Width of the outdoor-temperature bins the floor is read from, degC.
# Narrow enough that the coldest bins really are cold, wide enough that
# each holds enough hours to average.
COLD_BIN_WIDTH_C = 2.5

# Bins with fewer hours than this are ignored -- the coldest bin of the
# year often holds a handful of hours and its mean is noise.
MIN_BIN_HOURS = 100

# How many of the coldest surviving bins the floor averages over. More
# than one so a single unusual cold snap cannot set the bound; few
# enough that the average stays at the cold end of the year.
N_COLD_BINS = 3

# The bound is this multiple of the measured floor.
#
# A bound's job is to exclude the physically impossible, not to pin the
# answer, so this is deliberately generous. It also has to absorb a real
# modelling effect: the ventilation term is signed, so in cold weather it
# SUBTRACTS load, and the constant parameter must be larger than the
# observed floor to compensate. Measured, the constant the data demands
# is 1.7 to 3.2 times the observed floor:
#
#     Fox_education_Claude    floor 166 W/m2   demanded 315.6   1.90x
#     Bull_education_Luke     floor  69 W/m2   demanded 120.0   1.74x
#     Hog_education_Cathleen  floor  33 W/m2   demanded 104.0   3.15x
#
# 4.0 clears the largest observed case with headroom. A fit that still
# pins against a bound derived this way is making a strong statement:
# the building needs four times more constant load than it demonstrably
# draws when the weather contributes nothing, which is not a bounds
# problem any more.
INTERNAL_GAIN_BOUND_MULTIPLIER = 4.0


def cold_weather_floor_w_per_m2(
    load_kw: pd.Series,
    t_outdoor_c: pd.Series,
    floor_area_m2: float,
    bin_width_c: float = COLD_BIN_WIDTH_C,
    min_bin_hours: int = MIN_BIN_HOURS,
    n_cold_bins: int = N_COLD_BINS,
) -> float:
    """Cooling load that survives the coldest hours, W/m2.

    Bins the load by outdoor temperature and averages the coldest bins
    that hold enough hours to mean anything. Envelope conduction and
    ventilation load both vanish as outdoor temperature approaches the
    setpoint, so what remains at the cold end is the constant term --
    read straight off the data, with no model and no fitted parameter
    involved.

    Read the whole binned profile before trusting one number from it: if
    the load keeps falling toward zero in the coldest bins rather than
    flattening, there is no constant term and this statistic is
    measuring the tail of the envelope curve instead.

    Args:
        load_kw: Measured cooling load, indexed by timestamp.
        t_outdoor_c: Outdoor dry-bulb aligned to `load_kw`.
        floor_area_m2: Conditioned floor area, for the normalisation.
        bin_width_c: Temperature bin width.
        min_bin_hours: Bins holding fewer hours are ignored.
        n_cold_bins: How many of the coldest surviving bins to average.

    Returns:
        The floor, W/m2.

    Raises:
        ValueError: If `floor_area_m2` is not positive, if the inputs do
            not overlap, or if fewer than `n_cold_bins` bins survive the
            `min_bin_hours` cut -- a floor read from one thin bin is
            noise, and returning it silently would set a bound from it.
    """
    if floor_area_m2 <= 0.0:
        raise ValueError(f"floor_area_m2 must be > 0, got {floor_area_m2}")

    frame = pd.DataFrame({"load": load_kw, "t_out": t_outdoor_c}).dropna()
    if frame.empty:
        raise ValueError("load_kw and t_outdoor_c do not overlap")

    edges = np.arange(
        np.floor(frame["t_out"].min()),
        np.ceil(frame["t_out"].max()) + bin_width_c,
        bin_width_c,
    )
    binned = frame.groupby(
        pd.cut(frame["t_out"], edges, include_lowest=True), observed=True
    ).agg(hours=("load", "size"), mean_kw=("load", "mean"))
    populated = binned[binned["hours"] >= min_bin_hours]
    if len(populated) < n_cold_bins:
        raise ValueError(
            f"only {len(populated)} temperature bins hold at least "
            f"{min_bin_hours} hours (need {n_cold_bins}); this series is too "
            "short or too narrow in temperature to read a cold-weather floor"
        )

    floor_kw = float(populated["mean_kw"].head(n_cold_bins).mean())
    floor_w_per_m2 = 1000.0 * floor_kw / floor_area_m2
    logger.info(
        "cold-weather floor: %.1f kW = %.1f W/m2 (coldest %d bins of %d populated)",
        floor_kw,
        floor_w_per_m2,
        n_cold_bins,
        len(populated),
    )
    return floor_w_per_m2


def internal_gain_upper_bound(
    load_kw: pd.Series,
    t_outdoor_c: pd.Series,
    floor_area_m2: float,
    multiplier: float = INTERNAL_GAIN_BOUND_MULTIPLIER,
    **floor_kwargs: object,
) -> float:
    """Upper bound for `internal_gain_w_per_m2`, derived from the data.

    `multiplier` times `cold_weather_floor_w_per_m2`. Applied
    identically to every building -- the point of deriving it is that no
    building gets a bound chosen to suit it.

    Args:
        load_kw: Measured cooling load.
        t_outdoor_c: Outdoor dry-bulb aligned to `load_kw`.
        floor_area_m2: Conditioned floor area.
        multiplier: Headroom over the measured floor.
        **floor_kwargs: Passed to `cold_weather_floor_w_per_m2`.

    Returns:
        The upper bound, W/m2.

    Raises:
        ValueError: If `multiplier` is not greater than 1 -- a bound at
            or below the measured floor would forbid the load the
            building is observed to carry.
    """
    if multiplier <= 1.0:
        raise ValueError(
            f"multiplier must be > 1, got {multiplier}. At or below 1 the bound "
            "would exclude the constant load the building demonstrably draws "
            "when the weather contributes nothing."
        )
    floor = cold_weather_floor_w_per_m2(
        load_kw, t_outdoor_c, floor_area_m2, **floor_kwargs  # type: ignore[arg-type]
    )
    return multiplier * floor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Known-answer demo: a building with a 100 kW constant process load
    # plus a weather-driven term that switches off below 18 degC.
    index = pd.date_range("2016-01-01", periods=8784, freq="h", tz="UTC")
    position = np.arange(len(index), dtype=float)
    t_out = pd.Series(
        18.0 + 12.0 * np.sin(2 * np.pi * position / (24 * 365)) +
        4.0 * np.sin(2 * np.pi * position / 24),
        index=index,
    )
    load = pd.Series(100.0 + 20.0 * np.clip(t_out - 18.0, 0.0, None), index=index)

    floor = cold_weather_floor_w_per_m2(load, t_out, floor_area_m2=1000.0)
    logger.info("recovered floor %.1f W/m2 -- true constant is 100 kW / 1000 m2 = 100.0", floor)
    logger.info(
        "derived internal-gain upper bound: %.1f W/m2",
        internal_gain_upper_bound(load, t_out, floor_area_m2=1000.0),
    )

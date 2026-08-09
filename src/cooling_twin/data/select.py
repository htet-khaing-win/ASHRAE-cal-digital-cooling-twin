"""Building selection for the cooling twin project.

Applies the hard filters and soft-preference ranking defined in
04_DATA_CONTRACT.md SS2, and writes the selected buildings plus the
recorded reason for each selection to config/buildings.yaml.

Interface contract with L2.1's load.py (verified against the real file,
not assumed):
    load_metadata(path=METADATA_PATH) -> pd.DataFrame
        indexed by building_id; columns include site_id, primaryspaceusage,
        sqft, yearbuilt, timezone, and one Yes/NaN column per meter type.
    load_meter(meter_type, meters_dir=METERS_RAW_DIR) -> pd.DataFrame
        LONG format: columns timestamp, building_id, meter_reading.
    list_buildings_with_meter(metadata_df, meter_type) -> list[str]
        reads the metadata Yes/NaN flag column, not the raw meter file.

BDG2 reports floor area in sqft; the data contract (04_DATA_CONTRACT.md SS6)
specifies floor_area_m2, so the conversion happens once, here, at the
boundary -- nothing downstream ever sees sqft.

KNOWN LIMITATION -- read before rerunning from scratch:
This script's hard filters and soft scoring run purely against metadata.csv
and the meter completeness fraction. They do NOT run the L2.4 timezone
cross-correlation gate or a stuck-sensor check -- those require joining
weather data and are performed separately (see weather.py). Two exclusions
below (EXCLUDED_SITES, EXCLUDED_BUILDING_IDS) were discovered that way,
AFTER an initial run of this script, and are hardcoded here so a fresh run
does not silently reselect a combination already known to fail those later
checks. This is a stopgap, not a fix: a rerun of this script alone still
cannot discover a *new* timezone or stuck-sensor problem in a *different*
candidate. Folding L2.4 into this script's ranking loop is a reasonable
M9 (production packaging) task, not done here to keep L2.2 and L2.4 as
separate, teachable steps. See 07_PROGRESS.md ADR-004, ADR-005.

Run:
    PYTHONPATH=src python3 -m cooling_twin.data.select
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from cooling_twin.data.load import (
    METADATA_PATH,
    METERS_RAW_DIR,
    list_buildings_with_meter,
    load_metadata,
    load_meter,
)

logger = logging.getLogger(__name__)

# --- Unit conversion -----------------------------------------------------
# BDG2 metadata ships floor area in sqft; the data contract requires m2.
# Exact factor, not a rounded approximation.
SQFT_TO_M2 = 0.09290304

# --- Hard filter thresholds ------------------------------------------------
# Every number here traces back to 04_DATA_CONTRACT.md SS2.
MAX_MISSING_FRACTION = 0.10  # missing data < 10% per meter
REQUIRED_YEARS = (2016, 2017)

# --- Known exclusions from post-selection validation ------------------------
# These were NOT caught by the hard filters above -- they only surfaced
# after running L2.4's timezone cross-correlation gate and a stuck-sensor
# check against the top-ranked candidates (a step this script does not
# perform itself; see the module docstring). Recorded here so a fresh run
# of this script does not silently reproduce a selection already known to
# fail downstream. See 07_PROGRESS.md ADR-004 / ADR-005.
EXCLUDED_SITES: set[str] = {
    "Moose",  # ADR-005: 3/3 independently-tested Moose buildings failed
              # the L2.4 timezone gate with consistent negative peak lag
              # (-5h, -4h, -2h) and an identical 684-row weather join gap
              # -- a site-level weather file fault, not a per-building one.
}
EXCLUDED_BUILDING_IDS: set[str] = {
    "Hog_office_Darline",  # ADR-004: empirically flattest load shape
                            # candidate (cv=0.054), but its 7h identical-
                            # value run is a C3 stuck-sensor artifact, not
                            # genuine equipment-driven load.
}

# --- Soft preference weights ------------------------------------------------
PREFERRED_USES = {"Office", "Education"}
SOFT_WEIGHT_PREFERRED_USE = 2.0
SOFT_WEIGHT_YEAR_BUILT_PRESENT = 1.0
SOFT_WEIGHT_CLEANLINESS = 1.0  # proxy for "few flatline/zero streaks"


@dataclass(frozen=True)
class BuildingCandidate:
    """One building's filter result and soft score, with reasons recorded."""

    building_id: str
    site_id: str
    primary_use: str
    floor_area_m2: float
    year_built: int | None
    missing_frac_chilledwater: float
    missing_frac_electricity: float
    passed_hard_filters: bool
    hard_filter_failures: list[str] = field(default_factory=list)
    soft_score: float = 0.0
    soft_reasons: list[str] = field(default_factory=list)


def _missing_fraction_by_building(
    meter_long: pd.DataFrame, years: tuple[int, ...]
) -> pd.Series:
    """Fraction of missing hourly readings per building, across given years.

    meter_long is long-format (timestamp, building_id, meter_reading) from
    load_meter(). melt() preserves one row per (timestamp, building_id) pair
    that existed in the wide CSV's grid, with NaN where a reading is absent
    -- so isna().mean() per building over the year-filtered rows is exactly
    the missing fraction.

    Args:
        meter_long: Long-format meter DataFrame.
        years: Years to include in the denominator.

    Returns:
        Series indexed by building_id, missing fraction in [0, 1]. A
        building entirely absent from meter_long simply won't appear here --
        callers must handle that with a fallback default.
    """
    year_mask = meter_long["timestamp"].dt.year.isin(years)
    subset = meter_long.loc[year_mask, ["building_id", "meter_reading"]].copy()
    subset["is_missing"] = subset["meter_reading"].isna()
    return subset.groupby("building_id")["is_missing"].mean()


def _year_presence_by_building(meter_long: pd.DataFrame, year: int) -> pd.Series:
    """Whether each building has at least one non-null reading in `year`."""
    year_mask = meter_long["timestamp"].dt.year == year
    subset = meter_long.loc[year_mask, ["building_id", "meter_reading"]].copy()
    subset["is_present"] = subset["meter_reading"].notna()
    return subset.groupby("building_id")["is_present"].any()


def _apply_hard_filters(
    building_id: str,
    site_id: str,
    primary_use: object,
    floor_area_m2: float,
    missing_cw: float,
    missing_elec: float,
    has_2016: bool,
    has_2017: bool,
) -> list[str]:
    """Return failed hard filters for one building; empty list = passed.

    Checks the known-exclusion lists (EXCLUDED_SITES, EXCLUDED_BUILDING_IDS)
    first, before the data-contract filters, so a rerun of this script
    cannot silently reselect a building or site already known to fail the
    downstream L2.4 timezone gate or a stuck-sensor check.
    """
    failures: list[str] = []
    if site_id in EXCLUDED_SITES:
        failures.append(f"site {site_id} excluded -- see ADR-005 in 07_PROGRESS.md")
    if building_id in EXCLUDED_BUILDING_IDS:
        failures.append(
            f"building {building_id} excluded -- see ADR-004 in 07_PROGRESS.md"
        )
    if missing_cw > MAX_MISSING_FRACTION:
        failures.append(f"chilledwater missing {missing_cw:.1%} > {MAX_MISSING_FRACTION:.0%}")
    if missing_elec > MAX_MISSING_FRACTION:
        failures.append(f"electricity missing {missing_elec:.1%} > {MAX_MISSING_FRACTION:.0%}")
    if pd.isna(floor_area_m2):
        failures.append("floor_area_m2 missing (sqft absent in metadata)")
    if not primary_use or (isinstance(primary_use, float) and pd.isna(primary_use)):
        failures.append("primaryspaceusage missing")
    if not (has_2016 and has_2017):
        failures.append("data not present in both 2016 and 2017")
    return failures


def _soft_score(
    primary_use: object, year_built: int | None, missing_cw: float
) -> tuple[float, list[str]]:
    """Compute the soft preference score and the reason behind each point."""
    score = 0.0
    reasons: list[str] = []

    if primary_use in PREFERRED_USES:
        score += SOFT_WEIGHT_PREFERRED_USE
        reasons.append(f"primary_use={primary_use} (simple occupancy pattern)")

    if year_built is not None:
        score += SOFT_WEIGHT_YEAR_BUILT_PRESENT
        reasons.append(f"year_built={year_built} present")

    # Real flatline detection doesn't exist until L3.1-L3.3. Missing-hours
    # fraction is used as an explicit, named proxy in the meantime.
    cleanliness_bonus = SOFT_WEIGHT_CLEANLINESS * (1.0 - missing_cw / MAX_MISSING_FRACTION)
    score += cleanliness_bonus
    reasons.append(
        f"chilledwater missing only {missing_cw:.1%} "
        "(cleanliness proxy, not a real flatline check yet)"
    )

    return score, reasons


def score_candidates(
    metadata_path: Path = METADATA_PATH,
    meters_dir: Path = METERS_RAW_DIR,
) -> list[BuildingCandidate]:
    """Run every metadata row through the hard filters and soft scoring.

    Args:
        metadata_path: Path to metadata.csv.
        meters_dir: Directory containing the raw meter CSVs.

    Returns:
        Every candidate, passed or not -- rejections stay auditable too.
    """
    metadata = load_metadata(metadata_path)
    cw_long = load_meter("chilledwater", meters_dir)
    elec_long = load_meter("electricity", meters_dir)

    cw_buildings = set(list_buildings_with_meter(metadata, "chilledwater"))
    elec_buildings = set(list_buildings_with_meter(metadata, "electricity"))

    missing_cw_by_building = _missing_fraction_by_building(cw_long, REQUIRED_YEARS)
    missing_elec_by_building = _missing_fraction_by_building(elec_long, REQUIRED_YEARS)
    has_2016_by_building = _year_presence_by_building(cw_long, 2016)
    has_2017_by_building = _year_presence_by_building(cw_long, 2017)

    candidates: list[BuildingCandidate] = []
    for building_id, row in metadata.iterrows():
        primary_use = row.get("primaryspaceusage")
        year_built_raw = row.get("yearbuilt")
        year_built = int(year_built_raw) if pd.notna(year_built_raw) else None
        sqft = row.get("sqft")
        floor_area_m2 = float(sqft) * SQFT_TO_M2 if pd.notna(sqft) else float("nan")
        site_id = row.get("site_id", "")

        if building_id not in cw_buildings or building_id not in elec_buildings:
            missing_meters = [
                m for m, present in
                (("chilledwater", building_id in cw_buildings),
                 ("electricity", building_id in elec_buildings))
                if not present
            ]
            candidates.append(
                BuildingCandidate(
                    building_id=building_id,
                    site_id=site_id,
                    primary_use=primary_use or "",
                    floor_area_m2=floor_area_m2,
                    year_built=year_built,
                    missing_frac_chilledwater=1.0,
                    missing_frac_electricity=1.0,
                    passed_hard_filters=False,
                    hard_filter_failures=[
                        f"{m} meter absent (metadata flag)" for m in missing_meters
                    ],
                )
            )
            continue

        missing_cw = float(missing_cw_by_building.get(building_id, 1.0))
        missing_elec = float(missing_elec_by_building.get(building_id, 1.0))
        has_2016 = bool(has_2016_by_building.get(building_id, False))
        has_2017 = bool(has_2017_by_building.get(building_id, False))

        failures = _apply_hard_filters(
            building_id, site_id, primary_use, floor_area_m2,
            missing_cw, missing_elec, has_2016, has_2017,
        )
        passed = len(failures) == 0
        soft_score, soft_reasons = (
            (0.0, []) if not passed else _soft_score(primary_use, year_built, missing_cw)
        )

        candidates.append(
            BuildingCandidate(
                building_id=building_id,
                site_id=site_id,
                primary_use=primary_use or "",
                floor_area_m2=floor_area_m2,
                year_built=year_built,
                missing_frac_chilledwater=missing_cw,
                missing_frac_electricity=missing_elec,
                passed_hard_filters=passed,
                hard_filter_failures=failures,
                soft_score=soft_score,
                soft_reasons=soft_reasons,
            )
        )

    logger.info(
        "Scored %d candidates, %d passed hard filters",
        len(candidates),
        sum(c.passed_hard_filters for c in candidates),
    )
    return candidates


def select_buildings(
    candidates: list[BuildingCandidate],
    n_generalisation: int = 2,
    require_distinct_sites: bool = True,
) -> dict[str, list[BuildingCandidate]]:
    """Rank passing candidates and split into primary / generalisation roles.

    Args:
        candidates: Output of score_candidates().
        n_generalisation: How many additional buildings to select.
        require_distinct_sites: If True, prefer generalisation buildings from
            sites other than the primary's and each other's, since the role
            exists to test cross-site transfer (04_DATA_CONTRACT.md SS2).
            Falls back to same-site candidates, with a logged warning, only
            if too few distinct sites pass the hard filters.

    Returns:
        {"primary": [...], "generalisation": [...]}

    Raises:
        ValueError: If fewer than 1 + n_generalisation candidates pass.
    """
    passing = sorted(
        (c for c in candidates if c.passed_hard_filters),
        # Deterministic tie-break: soft_score desc, then the continuous
        # cleanliness proxy asc (finer-grained than the rounded score),
        # then building_id asc as a final, arbitrary-but-stable key.
        # Omitting a key here is what let ties fall back to metadata.csv's
        # row order -- which is grouped by site -- last time.
        key=lambda c: (-c.soft_score, c.missing_frac_chilledwater, c.building_id),
    )
    required = 1 + n_generalisation
    if len(passing) < required:
        raise ValueError(
            f"Only {len(passing)} candidates passed hard filters; "
            f"need at least {required} (1 primary + {n_generalisation} generalisation)."
        )

    primary = passing[0]
    generalisation: list[BuildingCandidate] = []
    used_sites = {primary.site_id}

    for c in passing[1:]:
        if len(generalisation) >= n_generalisation:
            break
        if require_distinct_sites and c.site_id in used_sites:
            continue
        generalisation.append(c)
        used_sites.add(c.site_id)

    if len(generalisation) < n_generalisation:
        chosen_ids = {primary.building_id} | {c.building_id for c in generalisation}
        remaining = [c for c in passing[1:] if c.building_id not in chosen_ids]
        shortfall = n_generalisation - len(generalisation)
        logger.warning(
            "Only %d distinct sites available among passing candidates; "
            "filling %d generalisation slot(s) with same-site buildings. "
            "This weakens the cross-site transfer claim -- record it in "
            "07_PROGRESS.md.",
            len(used_sites),
            shortfall,
        )
        generalisation.extend(remaining[:shortfall])

    return {"primary": [primary], "generalisation": generalisation}


def write_buildings_yaml(selection: dict[str, list[BuildingCandidate]], out_path: Path) -> None:
    """Write config/buildings.yaml with the selected buildings and their reasons.

    Args:
        selection: Output of select_buildings().
        out_path: Destination, e.g. config/buildings.yaml.
    """
    payload: dict[str, object] = {}
    for role, items in selection.items():
        payload[role] = [
            {
                "building_id": c.building_id,
                "site_id": c.site_id,
                "primary_use": c.primary_use,
                "floor_area_m2": round(c.floor_area_m2, 1),
                "missing_pct_chilledwater": round(c.missing_frac_chilledwater * 100, 2),
                "soft_score": round(c.soft_score, 2),
                "reasons": c.soft_reasons,
            }
            for c in items
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    candidates = score_candidates()
    selection = select_buildings(candidates, n_generalisation=2)
    write_buildings_yaml(selection, Path("config/buildings.yaml"))

    for role, items in selection.items():
        for c in items:
            print(f"{role}: {c.building_id} (site={c.site_id}, score={c.soft_score:.2f})")
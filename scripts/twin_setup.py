"""Shared M8 setup: turn a calibration artifact into a runnable twin.

Three scripts in this module's directory (`compare_correlation_intervention.py`,
`run_counterfactual.py`, `validate_intervals.py`) all need the same five
steps -- read the frozen parameters, load the building-year, build the
twin, size the plant, and simulate the baseline. Duplicating those five
steps three times is how three scripts end up quietly disagreeing about
which parameter set is "the" calibrated one.

THE TWO STRUCTURAL PROTECTIONS CARRIED FROM M7, unchanged:

  1. ADR-002. Training year only. Nothing in M8 touches 2017. A
     counterfactual is not a validation, so there is nothing a test year
     could confirm about it -- opening the test set for a what-if would
     spend the project's scarcest asset on a question it cannot answer.
  2. The optimiser is not imported. Parameters are READ and frozen.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import IncompatibleArtifactError, frozen_parameters  # noqa: E402
from run_calibration import load_training_data  # noqa: E402

from cooling_twin.models.plant import (  # noqa: E402
    DEFAULT_PLANT_CONFIG_PATH,
    PlantParams,
    build_plant_params,
    load_plant_config,
)
from cooling_twin.models.rc import DEFAULT_SUPPLY_HUMIDITY_RATIO  # noqa: E402
from cooling_twin.twin.counterfactual import CalibratedTwin  # noqa: E402

logger = logging.getLogger("twin_setup")

BUILDINGS_PATH = Path("config/buildings.yaml")

# The two buildings whose model holds ASHRAE G14 on the held-out year.
# `negative_case` is excluded by default for the same reason M7 excluded
# it from the hybrid: ADR-015 established that Hog_education_Cathleen's
# error is the inverse model's clip at zero binding on 4.95% of the
# year, and a counterfactual run on a model with a known structural
# defect produces a saving estimate that is a statement about the
# defect.
DEFAULT_GROUPS = ("primary", "generalisation")
NEGATIVE_CASE_GROUP = "negative_case"

__all__ = [
    "BUILDINGS_PATH",
    "DEFAULT_GROUPS",
    "NEGATIVE_CASE_GROUP",
    "IncompatibleArtifactError",
    "TwinBundle",
    "load_twin",
    "selected_buildings",
]


@dataclass(frozen=True)
class TwinBundle:
    """One building's twin, plant, measurements and index, in one object.

    Frozen for the same reason every other container in this project is:
    three scripts share it, and a scenario that could swap the plant or
    the measurements out from under a later panel would produce a
    comparison of two different things under one heading.

    Attributes:
        building_id: BDG2 identifier.
        role: Which group in `config/buildings.yaml` it came from.
        twin: The calibrated twin, ready to intervene on.
        plant: The plant serving it, sized from the measured peak.
        measured_kw: Measured cooling load, kW.
        index: The timestamp index of both series.
        artifact_name: The calibration artifact the parameters came from.
        frame: The full cleaned/joined frame, for scripts that need
            drivers beyond the four the twin holds.
    """

    building_id: str
    role: str
    twin: CalibratedTwin
    plant: PlantParams
    measured_kw: np.ndarray
    index: pd.DatetimeIndex
    artifact_name: str
    frame: pd.DataFrame


def selected_buildings(path: Path, groups: tuple[str, ...]) -> list[dict[str, str]]:
    """Buildings to run, in report order.

    Args:
        path: `config/buildings.yaml`.
        groups: Which roles to include.

    Returns:
        `[{"building_id": ..., "site_id": ..., "role": ...}]`.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If no primary building is declared.
    """
    if not path.exists():
        raise FileNotFoundError(f"building selection not found at {path}")
    selection = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not selection.get("primary"):
        raise ValueError(f"{path} declares no primary building")
    return [
        {"building_id": entry["building_id"], "site_id": entry["site_id"], "role": role}
        for role in groups
        for entry in (selection.get(role) or [])
    ]


def load_twin(
    building: dict[str, str],
    config: dict[str, Any],
    artifacts: Path,
    plant_config_path: Path | None = None,
) -> TwinBundle:
    """Read one building's frozen calibration and build its twin and plant.

    Args:
        building: `{"building_id", "site_id", "role"}`.
        config: Parsed `config/calibration.yaml`.
        artifacts: Directory holding the calibration artifacts.
        plant_config_path: Override for `config/plant.yaml`.

    Returns:
        A `TwinBundle`.

    Raises:
        FileNotFoundError: If the building has no calibration artifact.
        IncompatibleArtifactError: If none matches the current model.
        ValueError: If the data or the artifact is unusable.
    """
    building_id, site_id = building["building_id"], building["site_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])

    parameters, artifact_name = frozen_parameters(artifacts, building_id, train_year, names)
    frame, floor_area_m2 = load_training_data(building_id, site_id, train_year)

    twin = CalibratedTwin(
        t_seconds=(frame.index - frame.index[0]).total_seconds().to_numpy(dtype=float),
        t_ambient_c=frame["airTemperature"].to_numpy(dtype=float),
        humidity_ratio=frame["humidity_ratio"].to_numpy(dtype=float),
        wet_bulb_c=frame["wet_bulb_c"].to_numpy(dtype=float),
        floor_area_m2=floor_area_m2,
        parameters=parameters,
        envelope_capacity_ratio=float(config["fixed"]["envelope_capacity_ratio"]),
        ceiling_height_m=float(config["fixed"]["ceiling_height_m"]),
        supply_humidity_ratio=float(
            config.get("ventilation", {}).get(
                "supply_humidity_ratio", DEFAULT_SUPPLY_HUMIDITY_RATIO
            )
        ),
        building_id=building_id,
        year=train_year,
    )

    measured_kw = frame["load_kwh"].to_numpy(dtype=float)
    plant_config = load_plant_config(
        DEFAULT_PLANT_CONFIG_PATH if plant_config_path is None else plant_config_path
    )
    # Sized from the MEASURED peak, not the modelled one: the plant that
    # was installed had to serve what the meter recorded, and sizing off
    # the model would let a model error resize the building's plant.
    plant = build_plant_params(float(measured_kw.max()), plant_config)

    logger.info(
        "%s: twin built from %s (%d h, mean measured %.0f kW, peak %.0f kW)",
        building_id,
        artifact_name,
        measured_kw.size,
        float(measured_kw.mean()),
        float(measured_kw.max()),
    )
    return TwinBundle(
        building_id=building_id,
        role=building["role"],
        twin=twin,
        plant=plant,
        measured_kw=measured_kw,
        index=frame.index,
        artifact_name=artifact_name,
        frame=frame,
    )

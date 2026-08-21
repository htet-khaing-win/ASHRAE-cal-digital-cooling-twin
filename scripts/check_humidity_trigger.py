"""Is Claude's high-temperature deficit carried by the humid hours? (Q11)

    python scripts/check_humidity_trigger.py

L7.1c measured Fox_education_Claude's residual as a hockey stick: flat
below ~16 degC, then +54.4 kW/K up to 43 degC. Both weather parameters
are already at their ceilings (`vent_flow_kg_per_s` 179.60 of 180.0,
`ua_envelope_w_per_m2k` 3.00 of 3.0), and both of the model's weather
terms are LINEAR in the indoor-outdoor difference -- so raising either
adds slope everywhere, including the shoulder band where the model
already over-predicts by 557 kW. The missing term must therefore switch
on with temperature or with humidity rather than scale with deltaT
throughout.

Q11 names the first suspect: the latent term's TRIGGER.

    latent_w = vent_flow * h_fg * max(w_outdoor - w_supply, 0)

`w_supply` is a stated assumption fixed at 0.0092 kg/kg (ADR-011, never
fitted), so the latent term is EXACTLY ZERO for every hour drier than
9.2 g/kg. `config/calibration.yaml` already records that this assumption
is known to be imperfect -- L6.7b found a humidity effect at Theodore in
the 20-25 degC band, where a 12.8 degC coil predicts no latent load at
all.

THE TEST. The residual is what remains AFTER the latent term has been
applied. So if the implemented term were correctly sized and correctly
triggered, humidity should be gone from the residual. Using the L7.2
matched split -- bin on outdoor temperature, remove the within-bin
temperature trend, split each bin at its median humidity -- asks whether
humid hours still run above dry ones at the SAME outdoor temperature.

Run separately on the upper arm and the lower region, because Q11's
question is specifically about the upper arm. An effect that is present
everywhere is a mis-sized term; one confined to the hot hours is a
mis-triggered one; and no effect anywhere sends the search back to the
sensible path.

2016 only. The optimiser is not imported; parameters are read frozen.
Luke and Cathleen are included because a humidity effect that appears on
every building is a statement about the model, not about Claude.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_residuals import (  # noqa: E402
    IncompatibleArtifactError,
    frozen_parameters,
    selected_buildings,
)
from investigate_ushape import year_residual  # noqa: E402
from run_calibration import load_config  # noqa: E402

from cooling_twin.analysis.residual import (  # noqa: E402
    TEMPERATURE_BIN_WIDTH_K,
    Binning,
    effective_sample_size,
    matched_band_split,
    residual_profile,
)
from cooling_twin.models.rc import (  # noqa: E402
    DEFAULT_SUPPLY_HUMIDITY_RATIO,
    H_FG_J_PER_KG,
)

logger = logging.getLogger("check_humidity_trigger")

DEFAULT_CONFIG_PATH = Path("config/calibration.yaml")
BUILDINGS_PATH = Path("config/buildings.yaml")
DEFAULT_FIGURE_PATH = Path("reports/figures/l7_2_humidity_trigger.png")

# Hours required on each side before a region is split separately. The
# upper arm of a hockey stick is a minority of the year, and splitting a
# region that holds a few hundred hours in two, then splitting each
# temperature bin again at its median, exhausts the data quickly.
MIN_REGION_HOURS = 800

DRY_COLOUR = "#2171b5"
HUMID_COLOUR = "#b2182b"
TRIGGER_COLOUR = "#525252"

_G_PER_KG = 1000.0
_WATTS_PER_KW = 1000.0


def turning_point(
    residual_kw: npt.NDArray[np.float64], t_ambient_c: npt.NDArray[np.float64]
) -> float:
    """Outdoor temperature at which the binned residual bottoms out.

    Assumption-free: the coldest bin mean, not the vertex of a fitted
    parabola. L7.1c reports both and they disagree on Claude (16.2 degC
    binned against a 16.8 degC vertex), so the one used to SPLIT the
    data is the one that makes no functional assumption.
    """
    profile = residual_profile(
        residual_kw,
        t_ambient_c,
        name="outdoor_dry_bulb",
        unit="degC",
        binning=Binning.FIXED_WIDTH,
        normaliser_kw=1.0,
        width=TEMPERATURE_BIN_WIDTH_K,
    )
    return float(profile.centres[int(np.argmin(profile.means))])


def latent_activity(
    humidity_ratio_kg_per_kg: npt.NDArray[np.float64],
    predicted_kw: npt.NDArray[np.float64],
    vent_flow_kg_per_s: float,
    supply_humidity_ratio: float,
) -> dict[str, float]:
    """How much of the year the model's latent term is switched on.

    Recomputed from the same expression `inverse_cooling_load` uses
    rather than read back out of the simulation, because the simulation
    returns only the total. If this ever disagrees with `rc.py` the
    disagreement is the finding, so the formula is written out here in
    full rather than approximated.

    Args:
        humidity_ratio_kg_per_kg: Outdoor humidity ratio.
        predicted_kw: The model's total predicted load, for the share.
        vent_flow_kg_per_s: Calibrated ventilation flow.
        supply_humidity_ratio: The stated supply condition, 0.0092.

    Returns:
        Activity fractions and the latent load itself.
    """
    excess = np.maximum(humidity_ratio_kg_per_kg - supply_humidity_ratio, 0.0)
    latent_kw = vent_flow_kg_per_s * H_FG_J_PER_KG * excess / _WATTS_PER_KW
    active = excess > 0.0
    return {
        "supply_humidity_ratio_g_per_kg": supply_humidity_ratio * _G_PER_KG,
        "hours_latent_active_pct": float(active.mean() * 100.0),
        "latent_mean_kw_when_active": (
            float(latent_kw[active].mean()) if active.any() else 0.0
        ),
        "latent_mean_kw_over_year": float(latent_kw.mean()),
        "latent_share_of_predicted_pct": float(
            latent_kw.sum() / predicted_kw.sum() * 100.0
        ),
        "outdoor_humidity_p50_g_per_kg": float(
            np.median(humidity_ratio_kg_per_kg) * _G_PER_KG
        ),
        "outdoor_humidity_p95_g_per_kg": float(
            np.percentile(humidity_ratio_kg_per_kg, 95) * _G_PER_KG
        ),
    }


def split_region(
    residual: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    humidity: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    ratio: float,
    label: str,
) -> dict[str, Any] | None:
    """Matched humidity split over one temperature region, or None."""
    if mask.sum() < MIN_REGION_HOURS:
        logger.info("%s: only %d hours, not split", label, int(mask.sum()))
        return None
    split = matched_band_split(
        residual[mask],
        temperature[mask],
        humidity[mask],
        control="outdoor_dry_bulb",
        probe="humidity_ratio",
        control_width=TEMPERATURE_BIN_WIDTH_K,
        effective_sample_ratio=ratio,
    )
    return {
        "region": label,
        "hours": int(mask.sum()),
        "weighted_difference_kw": split.weighted_difference_kw,
        "weighted_difference_sem_kw": split.weighted_difference_sem_kw,
        "humid_hours_run_higher": split.probe_raises_residual,
        "bins": int(split.centres.size),
        "_split": split,
    }


def check(
    building: dict[str, str], config: dict[str, Any], artifacts: Path
) -> dict[str, Any]:
    """Run Q11's test for one building on the training year."""
    building_id = building["building_id"]
    train_year = int(config["train_year"])
    names = tuple(config["parameters"])
    parameters, artifact_name = frozen_parameters(
        artifacts, building_id, train_year, names
    )

    series = year_residual(building, config, parameters, train_year)
    residual = series["residual_kw"]
    temperature = series["outdoor_dry_bulb"]
    humidity = series["humidity_ratio"] / _G_PER_KG  # back to kg/kg
    ratio = effective_sample_size(residual) / residual.size

    turn = turning_point(residual, temperature)
    supply = float(
        config.get("ventilation", {}).get(
            "supply_humidity_ratio", DEFAULT_SUPPLY_HUMIDITY_RATIO
        )
    )

    regions = [
        split_region(
            residual, temperature, humidity, np.ones_like(temperature, bool), ratio,
            "whole year",
        ),
        split_region(
            residual, temperature, humidity, temperature >= turn, ratio,
            f"upper arm (>= {turn:.1f} degC)",
        ),
        split_region(
            residual, temperature, humidity, temperature < turn, ratio,
            f"lower region (< {turn:.1f} degC)",
        ),
    ]

    return {
        "building_id": building_id,
        "role": building["role"],
        "calibration_artifact": artifact_name,
        "year": train_year,
        "hours": int(residual.size),
        "turning_point_degC": turn,
        "vent_flow_kg_per_s": parameters["vent_flow_kg_per_s"],
        "effective_sample_ratio": ratio,
        "latent": latent_activity(
            humidity, series["predicted_kw"], parameters["vent_flow_kg_per_s"], supply
        ),
        "regions": [
            {k: v for k, v in region.items() if not k.startswith("_")}
            for region in regions
            if region is not None
        ],
        "_regions": [region for region in regions if region is not None],
        "_series": series,
        "_supply": supply,
    }


def plot_humidity_split(records: list[dict[str, Any]], path: Path) -> Path:
    """Residual against temperature, humid half versus dry half."""
    figure, axes = plt.subplots(
        1, len(records), figsize=(4.8 * len(records), 4.2), squeeze=False
    )
    for axis, record in zip(axes[0], records, strict=True):
        whole = next(
            region for region in record["_regions"] if region["region"] == "whole year"
        )
        split = whole["_split"]
        axis.plot(
            split.centres,
            split.means_low,
            marker="o",
            markersize=3,
            color=DRY_COLOUR,
            label="dry half of each bin",
        )
        axis.plot(
            split.centres,
            split.means_high,
            marker="o",
            markersize=3,
            color=HUMID_COLOUR,
            label="humid half of each bin",
        )
        axis.axvline(
            record["turning_point_degC"],
            color=TRIGGER_COLOUR,
            linestyle="--",
            linewidth=1,
            label=f"residual minimum {record['turning_point_degC']:.1f} degC",
        )
        axis.axhline(0.0, color=TRIGGER_COLOUR, linewidth=1)
        latent = record["latent"]
        axis.set_title(
            f"{record['building_id']}\n"
            f"latent term active {latent['hours_latent_active_pct']:.0f}% of hours "
            f"(w > {latent['supply_humidity_ratio_g_per_kg']:.1f} g/kg)",
            fontsize=9,
        )
        axis.set_xlabel("outdoor dry bulb, degC", fontsize=8)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    axes[0][0].set_ylabel(
        "residual after the within-bin\ntemperature trend is removed, kW", fontsize=9
    )
    figure.suptitle(
        "Q11 -- at MATCHED outdoor temperature, do humid hours still run above "
        "dry ones? Training year only."
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info("figure: %s", path)
    return path


def main() -> None:
    """Entry point: run Q11's humidity split on every building."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--buildings", type=Path, default=BUILDINGS_PATH)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE_PATH)
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    config = load_config(arguments.config)
    artifacts = Path(config["artifacts"]["directory"])

    records = []
    for building in selected_buildings(arguments.buildings):
        try:
            records.append(check(building, config, artifacts))
        except (FileNotFoundError, IncompatibleArtifactError) as error:
            logger.info("skipping %s: %s", building["building_id"], error)

    if not records:
        raise SystemExit("no building had a calibration artifact to analyse")

    latent_table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "vent flow kg/s": round(record["vent_flow_kg_per_s"], 1),
                "w_supply g/kg": round(
                    record["latent"]["supply_humidity_ratio_g_per_kg"], 1
                ),
                "outdoor w p50": round(
                    record["latent"]["outdoor_humidity_p50_g_per_kg"], 1
                ),
                "outdoor w p95": round(
                    record["latent"]["outdoor_humidity_p95_g_per_kg"], 1
                ),
                "latent ON %": round(record["latent"]["hours_latent_active_pct"], 1),
                "latent kW when on": round(
                    record["latent"]["latent_mean_kw_when_active"], 0
                ),
                "% of predicted": round(
                    record["latent"]["latent_share_of_predicted_pct"], 2
                ),
            }
            for record in records
        ]
    ).set_index("building")
    logger.info(
        "--- 1. how much of the year the model's LATENT TERM is switched on "
        "at all ---\n%s",
        latent_table.to_string(),
    )

    region_table = pd.DataFrame(
        [
            {
                "building": record["building_id"],
                "region": region["region"],
                "hours": region["hours"],
                "humid-dry kW": round(region["weighted_difference_kw"], 1),
                "+/- kW": round(region["weighted_difference_sem_kw"], 1),
                "bins": region["bins"],
                "humid higher?": region["humid_hours_run_higher"],
            }
            for record in records
            for region in record["_regions"]
        ]
    ).set_index(["building", "region"])
    logger.info(
        "--- 2. THE TEST: at MATCHED outdoor temperature, with the within-bin "
        "temperature trend removed ---\n%s",
        region_table.to_string(),
    )

    logger.info(
        "--- how to read this ---\n"
        "  The residual is what is left AFTER the latent term has been\n"
        "  applied. A humidity effect that is still there means the term as\n"
        "  implemented is wrong, not missing.\n"
        "    effect on the UPPER ARM only  -> the TRIGGER is wrong; the term\n"
        "                                     is inert where it is needed.\n"
        "    effect EVERYWHERE             -> the term is mis-SIZED.\n"
        "    effect NOWHERE                -> humidity is not the missing\n"
        "                                     term, and the sensible path\n"
        "                                     needs re-examining instead.\n"
        "  Table 1 is the mechanism table 2 would be explained by: a term\n"
        "  fixed at w_supply = 9.2 g/kg is exactly zero for every drier hour."
    )

    plot_humidity_split(records, arguments.figure)

    out_path = artifacts / f"humidity_trigger_{int(config['train_year'])}.json"
    out_path.write_text(
        json.dumps(
            {
                "year": int(config["train_year"]),
                "question": "Q11",
                "note": (
                    "Training year only. Parameters read frozen; the optimiser "
                    "is not imported. Latent activity is recomputed from the "
                    "same expression rc.inverse_cooling_load uses."
                ),
                "buildings": [
                    {k: v for k, v in record.items() if not k.startswith("_")}
                    for record in records
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("artifact: %s", out_path)


if __name__ == "__main__":
    main()

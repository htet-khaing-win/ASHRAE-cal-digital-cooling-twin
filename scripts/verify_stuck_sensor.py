# scripts/verify_stuck_sensor.py
from pathlib import Path
from cooling_twin.data.load import load_meter

FLATLINE_THRESHOLD_HOURS = 6  # C3 rule from 04_DATA_CONTRACT.md

def longest_flat_run(series):
    """Longest run of consecutive identical (non-NaN) values, in hours."""
    is_same = series.eq(series.shift()).fillna(False)
    run_id = (~is_same).cumsum()
    run_lengths = series.groupby(run_id).transform("size")
    return int(run_lengths.max()) if len(run_lengths) else 0


def check_building(building_id: str, meters_dir: Path) -> None:
    cw = load_meter("chilledwater", meters_dir)
    series = cw.loc[cw["building_id"] == building_id, "meter_reading"].dropna()
    longest = longest_flat_run(series)
    verdict = "STUCK SENSOR (fails C3)" if longest >= FLATLINE_THRESHOLD_HOURS else "clean"
    print(f"{building_id}: longest flat run = {longest}h -> {verdict}")


if __name__ == "__main__":
    METERS_DIR = Path.home() / "data" / "bdg2" / "data" / "meters" / "raw"
    for bid in ["Moose_education_Abbie", "Panther_education_Aurora"]:
        check_building(bid, METERS_DIR)
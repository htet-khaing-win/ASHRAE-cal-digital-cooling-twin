"""Interactive project dashboard.

    streamlit run dashboard/app.py

Replaces `scripts/build_dashboard.py`'s static HTML page. That page's
own module docstring said "the page is a SUMMARY, never a source" -- true
of every NUMBER on it, but the page itself was three fixed screenshots
wearing a dashboard's clothes: nothing on it re-rendered, the train/test
split was a fixed pair of numbers rather than a control, and the
predicted-vs-actual comparison was one static PNG of one week. M0's own
definition of what a twin is NOT lists "a dashboard" among the things
that only look like an interface -- this page exists so the project's own
presentation layer stops being an example of that.

Every number is either:
  (a) read verbatim from a `reports/calibration_runs/*.json` artifact, or
  (b) recomputed live by calling the same `cooling_twin` functions the
      project's own scripts call, on the same frozen parameters and the
      same BDG2 rows those scripts load.
See `dashboard/data.py` for exactly which is which, function by function.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# `streamlit run dashboard/app.py` executes this file as __main__ and puts
# only its own directory on sys.path, not the repo root -- so `import
# dashboard` (the package, for `dashboard.data`) needs the repo root added
# explicitly, the same pattern scripts/twin_setup.py uses for `scripts/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dashboard import data  # noqa: E402

st.set_page_config(page_title="Cooling Digital Twin", page_icon="\U0001f9ca", layout="wide")

_SETPOINT_MIN_C = -2.0
_SETPOINT_MAX_C = 2.0
_SETPOINT_STEP_C = 0.5


def render_banner() -> None:
    """The permanent, non-dismissable scope statement.

    Rendered on every page load, above the fold, with no expander or
    close control -- the task brief is explicit that this is not a
    footnote to be trimmed for visual balance.
    """
    st.error(f"**Scope.** {data.BANNER_TEXT}", icon="⚠️")


def render_negative_case_banner(building_id: str, roster_entry: dict[str, Any]) -> None:
    """Cathleen's ADR-014 status, shown loudly whenever she is selected.

    Args:
        building_id: The currently selected building.
        roster_entry: Her `config/buildings.yaml` `negative_case` record.
    """
    if roster_entry["role"] != "negative_case":
        return
    reasons = roster_entry.get("rejected_because", [])
    st.error(
        f"**{building_id} failed the ASHRAE G14 gate on the held-out year "
        "(ADR-014).** This building is shown as a negative case, not as a "
        "working twin. Every panel below still reflects its actual "
        "calibration and residual behaviour -- nothing is hidden or "
        "smoothed over to make the default view cleaner.",
        icon="\U0001f6ab",
    )
    with st.expander("Why it failed (from config/buildings.yaml)", expanded=True):
        for reason in reasons:
            st.markdown(f"- {reason}")


def render_gate_cards(gate_record: dict[str, Any] | None) -> None:
    """Train/test ASHRAE G14 verdict cards for the selected building.

    Args:
        gate_record: This building's entry from `gate_2017_opened.json`,
            or `None` if it has no gate record (should not happen for
            any of the three roster buildings, but the panel says so
            rather than crashing if a config drifts).
    """
    if gate_record is None:
        st.warning("No gate artifact entry for this building.")
        return
    cols = st.columns(4)
    train, test = gate_record["train"], gate_record["test"]
    cols[0].metric("Train (2016) CV(RMSE)", f"{train['cvrmse_pct']:.1f}%")
    cols[1].metric("Train NMBE", f"{train['nmbe_pct']:+.2f}%")
    cols[2].metric(
        "Test (2017) CV(RMSE)",
        f"{test['cvrmse_pct']:.1f}%",
        delta=f"{test['cvrmse_pct'] - train['cvrmse_pct']:+.1f} pp vs train",
        delta_color="inverse",
    )
    cols[3].metric("Test NMBE", f"{test['nmbe_pct']:+.2f}%")
    verdict = "PASSED" if test["passed"] else "FAILED"
    st.caption(
        f"ASHRAE G14 hourly gate on the held-out year: **{verdict}** "
        f"(limit CV(RMSE) ≤ 30%, |NMBE| ≤ 10%). "
        f"Relative improvement over the best baseline: "
        f"{gate_record['relative_improvement_pct']:+.1f}% "
        f"(source: reports/calibration_runs/gate_2017_opened.json)."
    )


def render_predicted_vs_actual(series: data.YearSeries, building_id: str, year: int) -> None:
    """The scrubbable predicted-vs-actual panel.

    A date-range picker rather than a fixed one-week PNG: every hour of
    the selected year is available, at any window width, and the chart
    re-renders on every change. Series are recomputed live in
    `dashboard/data.load_year_series` -- see that function's docstring
    for exactly how 2017 is scored (frozen 2016 parameters, spin-up
    discarded), matching `scripts/open_test_set.py`.

    Args:
        series: The building-year's measured/predicted series.
        building_id: For the caption.
        year: 2016 or 2017, for the caption and default window.
    """
    st.subheader("Predicted vs. actual cooling load")
    index = series.index
    min_date, max_date = index.min().date(), index.max().date()

    preset = st.radio(
        "Window",
        ("First 2 weeks", "Peak-load week", "Full year", "Custom range"),
        horizontal=True,
        key="window_preset",
    )
    if preset == "First 2 weeks":
        start, end = min_date, min(max_date, min_date + pd.Timedelta(days=14))
    elif preset == "Peak-load week":
        peak_ts = index[int(np.argmax(series.measured_kw))]
        start = max(min_date, (peak_ts - pd.Timedelta(days=3)).date())
        end = min(max_date, (peak_ts + pd.Timedelta(days=4)).date())
    elif preset == "Full year":
        start, end = min_date, max_date
    else:
        picked = st.date_input(
            "Date range",
            value=(min_date, min(max_date, min_date + pd.Timedelta(days=14))),
            min_value=min_date,
            max_value=max_date,
            key="custom_range",
        )
        # `st.date_input` returns a single date mid-pick, before the second
        # end of the range has been clicked -- hold the previous window
        # rather than crashing the whole page on a transient one-tuple.
        if isinstance(picked, tuple) and len(picked) == 2:
            start, end = picked
        else:
            start, end = min_date, min(max_date, min_date + pd.Timedelta(days=14))

    mask = (index.date >= start) & (index.date <= end)
    if not mask.any():
        st.info("No hours in the selected range.")
        return

    frame = pd.DataFrame(
        {
            "timestamp": index[mask],
            "Measured (meter)": series.measured_kw[mask],
            "Predicted (twin)": series.predicted_kw[mask],
        }
    ).melt("timestamp", var_name="series", value_name="kW")

    chart = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title="Hour"),
            y=alt.Y("kW:Q", title="Cooling load (kW)"),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Measured (meter)", "Predicted (twin)"],
                    range=["#4c78a8", "#e45756"],
                ),
            ),
            tooltip=["timestamp:T", "series:N", alt.Tooltip("kW:Q", format=".0f")],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(chart, width='stretch')

    n_hours = int(mask.sum())
    mean_abs_error = float(np.abs(series.measured_kw[mask] - series.predicted_kw[mask]).mean())
    st.caption(
        f"{building_id}, {year} ({'training' if year == data.TRAIN_YEAR else 'held-out test'} "
        f"year) -- {n_hours} hours shown, mean |error| {mean_abs_error:,.0f} kW. Predicted "
        "series recomputed live from the frozen calibrated parameters against BDG2 weather "
        "for this window (dashboard/data.load_year_series)."
    )


def render_hybrid_decomposition(roster: list[dict[str, str]], selected_building: str) -> None:
    """Physics/ML/unexplained split, every building, from the L7.3 artifact.

    Args:
        roster: The three-building roster.
        selected_building: Highlighted with full opacity; the other two
            are shown dimmed rather than omitted -- requirement 4 is
            explicit that this must not collapse to the primary building
            alone.
    """
    st.subheader("Error decomposition: physics vs. ML residual vs. unexplained")
    hybrid = data.load_artifact("hybrid_2016.json")
    rows = []
    missing: list[str] = []
    for entry in roster:
        record = data.artifact_building_record(hybrid, entry["building_id"])
        if record is None:
            # A building with no hybrid run must be NAMED, never just
            # dropped: the caption below promises the non-selected
            # buildings are dimmed rather than hidden, and an absent bar
            # for the CURRENTLY SELECTED building is the exact silent
            # empty-panel failure L7.5 was caught by (07_PROGRESS.md,
            # Session 025). Absent from the artifact is a fact about the
            # project, so it is reported as one.
            missing.append(entry["building_id"])
            continue
        out_of_fold = record["out_of_fold"]
        for component, value in (
            ("Physics", out_of_fold["physics_pct"]),
            ("ML residual (out-of-fold)", out_of_fold["ml_pct"]),
            ("Unexplained", out_of_fold["unexplained_pct"]),
        ):
            rows.append(
                {
                    "building_id": entry["building_id"],
                    "component": component,
                    "pct": value,
                    "selected": entry["building_id"] == selected_building,
                }
            )
    if missing:
        st.info(
            f"**No bar for {', '.join(missing)}** -- "
            "reports/calibration_runs/hybrid_2016.json has no record for "
            f"{'this building' if len(missing) == 1 else 'these buildings'}, so there is no "
            "measured physics/ML split to draw. Shown as absent rather than as a zero bar: "
            "an unmeasured decomposition is not a decomposition of zero.",
            icon="\U0001f4ed",
        )
    if not rows:
        return

    frame = pd.DataFrame(rows)

    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("building_id:N", title=None, sort=[e["building_id"] for e in roster]),
            y=alt.Y("pct:Q", title="Share of annual variance explained (%)", stack="zero"),
            color=alt.Color(
                "component:N",
                title=None,
                scale=alt.Scale(
                    domain=["Physics", "ML residual (out-of-fold)", "Unexplained"],
                    range=["#4c78a8", "#f2a900", "#b0b0b0"],
                ),
            ),
            opacity=alt.condition(alt.datum.selected, alt.value(1.0), alt.value(0.35)),
            tooltip=["building_id:N", "component:N", alt.Tooltip("pct:Q", format=".1f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width='stretch')
    st.caption(
        "2016, out-of-fold (5-fold, 168 h embargo) -- the ML share is measured on hours the "
        "residual model was not fitted on. Source: reports/calibration_runs/hybrid_2016.json. "
        "Every building the artifact covers is drawn; those other than the current selection "
        "are dimmed, not hidden. Any building the artifact does not cover is named above "
        "rather than silently omitted."
    )


def render_setpoint_what_if(
    building_id: str, train_series: data.YearSeries, roster_entry: dict[str, str]
) -> None:
    """The zone-setpoint slider, wired to `simulate_setpoint_change()`.

    ADR-002 governs this panel regardless of which year the train/test
    toggle above is showing: a counterfactual is only ever run on the
    TRAINING year's calibrated twin, because the held-out year has no
    ground truth for a world that was never recorded.

    Args:
        building_id: For the equifinality lookup and caption.
        train_series: The 2016 `YearSeries` -- always 2016, never the
            year currently selected in the toggle.
        roster_entry: Current building's roster record, for the
            negative-case caveat.
    """
    st.subheader("Setpoint what-if")
    st.caption(
        "Always simulated against the **2016 (training)** calibrated twin, per ADR-002: "
        "a counterfactual has no ground truth on any year, so it is never run against "
        "2017 regardless of which year is selected above."
    )
    if roster_entry["role"] == "negative_case":
        st.warning(
            f"{building_id} failed the G14 gate on 2017. A counterfactual run on a model "
            "with a known structural defect (ADR-015's zero-clip) produces a saving estimate "
            "that is a statement about the defect, not the building. Shown for completeness, "
            "not as a recommendation.",
            icon="⚠️",
        )

    delta_c = st.slider(
        "Zone setpoint change (K)",
        min_value=_SETPOINT_MIN_C,
        max_value=_SETPOINT_MAX_C,
        value=1.0,
        step=_SETPOINT_STEP_C,
        key="setpoint_slider",
        help="Positive = relax the setpoint (e.g. 24 -> 25 degC). Negative = tighten it.",
    )

    if delta_c == 0.0:
        st.info("0 K is the calibrated baseline -- move the slider to simulate an intervention.")
        return

    with st.spinner("Re-solving the twin under this intervention..."):
        result = data.compute_setpoint_what_if(building_id, train_series, delta_c)

    cols = st.columns(4)
    cols[0].metric(
        "Total plant electricity change",
        f"{result.total_change_pct:+.1f}%",
        help="Point estimate from simulate_setpoint_change() (L8.2): chiller + pump, "
        "same weather and plant, only the setpoint intervened on.",
    )
    cols[1].metric(
        "± block-bootstrap (90%)",
        f"[{result.bootstrap_lower_pct:+.1f}%, {result.bootstrap_upper_pct:+.1f}%]",
        help="Sampling variability of the annual-mean change, week-block bootstrap "
        "(preserves the residual's autocorrelation). Says nothing about model error.",
    )
    if result.parameter_ensemble_pct:
        lo, hi = min(result.parameter_ensemble_pct), max(result.parameter_ensemble_pct)
        cols[2].metric(
            f"Parameter-ensemble spread (n={result.n_parameter_sets})",
            f"[{lo:+.1f}%, {hi:+.1f}%]",
            help="Same intervention re-run on every behavioural parameter set from L6.8's "
            "equifinality study -- the term that dominates, and the only one that measures "
            "structural uncertainty rather than sampling noise.",
        )
    else:
        cols[2].metric("Parameter-ensemble spread", "n/a")
        cols[2].caption("No equifinality study on disk for this building.")
    cols[3].metric(
        "Conformal hourly band width (mean)",
        f"±{float((result.conformal_upper_kw - result.conformal_lower_kw).mean() / 2):,.0f} kW",
        help="90% distribution-free interval on the scenario's hourly cooling load "
        "(conformal_interval(), L8.3), from the twin's own residual on the calibration split.",
    )

    hours = train_series.index
    band_frame = pd.DataFrame(
        {
            "timestamp": hours,
            "scenario_kw": result.scenario_load_kw,
            "lower_kw": result.conformal_lower_kw,
            "upper_kw": result.conformal_upper_kw,
        }
    )
    window_days = st.slider(
        "Band preview window (days from year start)",
        min_value=1,
        max_value=30,
        value=7,
        key="band_window",
    )
    windowed = band_frame[
        band_frame["timestamp"] < band_frame["timestamp"].iloc[0] + pd.Timedelta(days=window_days)
    ]
    band = (
        alt.Chart(windowed)
        .mark_area(opacity=0.25, color="#e45756")
        .encode(x="timestamp:T", y="lower_kw:Q", y2="upper_kw:Q")
    )
    line = (
        alt.Chart(windowed)
        .mark_line(color="#e45756")
        .encode(x="timestamp:T", y=alt.Y("scenario_kw:Q", title="Cooling load under scenario (kW)"))
    )
    st.altair_chart((band + line).properties(height=260), width='stretch')
    st.caption(
        "No point estimate above is shown without an interval. Sources: "
        "cooling_twin.twin.counterfactual.simulate_setpoint_change, "
        "cooling_twin.twin.uncertainty.conformal_interval / block_bootstrap_ci, "
        "and reports/calibration_runs/equifinality_*.json."
    )


def main() -> None:
    """Assemble the page."""
    render_banner()
    st.title("Cooling Digital Twin -- Interactive Dashboard")

    roster = data.building_roster()
    roster_by_id = {entry["building_id"]: entry for entry in roster}
    gate = data.load_artifact("gate_2017_opened.json")

    top = st.columns([2, 1])
    with top[0]:
        building_id = st.selectbox(
            "Building",
            options=[entry["building_id"] for entry in roster],
            format_func=lambda bid: f"{bid} ({data.ROLE_LABELS[roster_by_id[bid]['role']]})",
        )
    with top[1]:
        year = st.radio(
            "Year",
            (data.TRAIN_YEAR, data.TEST_YEAR),
            format_func=lambda y: f"{y} ({'train' if y == data.TRAIN_YEAR else 'test, held-out'})",
            horizontal=True,
        )

    roster_entry = roster_by_id[building_id]
    render_negative_case_banner(building_id, roster_entry)

    st.caption(
        f"Role: **{data.ROLE_LABELS[roster_entry['role']]}** &nbsp;|&nbsp; "
        f"Site: {roster_entry['site_id']} &nbsp;|&nbsp; "
        "config/buildings.yaml"
    )
    render_gate_cards(data.artifact_building_record(gate, building_id))

    st.divider()
    series = data.load_year_series(building_id, roster_entry["site_id"], year)
    render_predicted_vs_actual(series, building_id, year)

    st.divider()
    render_hybrid_decomposition(roster, building_id)

    st.divider()
    train_series = (
        series if year == data.TRAIN_YEAR
        else data.load_year_series(building_id, roster_entry["site_id"], data.TRAIN_YEAR)
    )
    render_setpoint_what_if(building_id, train_series, roster_entry)


if __name__ == "__main__":
    main()

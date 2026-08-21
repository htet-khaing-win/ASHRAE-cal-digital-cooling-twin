"""Hybrid physics + ML -- learning the residual, not the target (L7.3).

L7.1 found WHERE each model's error lives. L7.2 proved it is not noise:
the share of residual variance surviving daily averaging is 0.34-0.80
against a white-noise 0.042, i.e. 8x to 19x. An error that survives
averaging 24 hours is systematic, and systematic means learnable.

This module learns it, and then states how much was learnt as a number:

    measured = physics + ML correction + unexplained

    physics %      variance of the measured load the calibrated 2R2C
                   model explains, relative to the annual-mean baseline
    ML %           the ADDITIONAL variance the residual model explains
                   ON HOURS IT DID NOT SEE
    unexplained %  what is left

Those three sum to 100 by construction, which is the point: "physics
explains 84%" is a quantitative interpretability claim that a reviewer
can check. A SHAP bar chart is not -- it ranks features inside whatever
model you happened to fit, and says nothing about how much of the
building the model got.

WHY LEARN THE RESIDUAL RATHER THAN THE LOAD. A gradient-boosted tree
ensemble cannot extrapolate: outside the range of its training features
every tree returns its edge leaf, so the prediction goes FLAT. Train it
on the load directly and the twin predicts a constant for any weather
hotter than 2016's hottest hour -- which is precisely the weather M8's
counterfactuals and any climate-shifted scenario will ask about. Train
it on the RESIDUAL and the physics carries the trend out of range while
the ML correction decays to its edge value: the twin degrades to "the
calibrated physical model" rather than to "a constant".

WHY THE FEATURES ARE EXOGENOUS ONLY. No lagged load, no lagged residual,
no lagged meter reading -- see `FEATURE_NAMES`. Persistence features
would raise the ML share enormously and would make the hybrid useless
for the only question M8 asks: what would this building have drawn under
a setpoint it never ran. In the counterfactual world there is no
yesterday's meter reading to look up. A feature that does not exist in
the world where the model is used is not a feature.

WHY THE ML SHARE IS MEASURED OUT-OF-FOLD. A boosted ensemble fitted to
8,760 hours and scored on the same 8,760 hours will report a large ML
share whether or not it learnt anything transferable, because it can
memorise. Every ML number here comes from `expanding_window_folds`
(L6.7): fit on the past, score the block after it, and never score an
hour the model was fitted on. The in-sample decomposition is computed
too and reported alongside -- the GAP between the two is the
memorisation, and it is more informative than either number alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cooling_twin import SEED
from cooling_twin.analysis.residual import ResidualDiagnostics, residual_diagnostics
from cooling_twin.calibration.crossval import TimeFold, expanding_window_folds
from cooling_twin.calibration.metrics import cvrmse, nmbe

logger = logging.getLogger(__name__)

# Features the residual model is allowed to see. EXOGENOUS ONLY: every
# one of these exists in a counterfactual world where the building ran a
# setpoint it never ran (M8). Deliberately absent:
#
#   * lagged load / lagged residual -- persistence, unavailable in a
#     counterfactual, and it would dominate every other feature;
#   * month or day-of-year -- a tree given the calendar learns "August
#     is +500 kW on this building in this year", which is a memorised
#     year, not a mechanism, and it transfers to nothing;
#   * the physics prediction itself -- it is already a deterministic
#     function of these same drivers, so it adds little information
#     while letting the ensemble partially RESCALE the physics. The
#     decomposition below would then credit the ML with variance the
#     physics term has already been credited with, and the split stops
#     meaning anything. Include it if the goal is accuracy alone; not
#     when the deliverable is the attribution.
FEATURE_NAMES: tuple[str, ...] = (
    "outdoor_dry_bulb_c",
    "outdoor_dry_bulb_24h_mean_c",
    "humidity_ratio_g_per_kg",
    "hour_sin",
    "hour_cos",
    "is_weekend",
)

# Window for the lagged-weather feature, hours. Weather from the last
# day is exogenous (it is an input to the counterfactual, not an output
# of it), and it is the only way a memoryless tree ensemble can see the
# building's thermal mass at all -- the physics model carries that in
# its envelope state, the ML model has no state whatsoever.
WEATHER_WINDOW_HOURS = 24

# Hours required inside the window before the rolling mean is reported.
# Below this the "24-hour mean" is a mean of six hours wearing the same
# column name. NaN is returned instead, and the ensemble handles it
# natively -- see `default_model_factory`.
WEATHER_WINDOW_MIN_PERIODS = 12

# Hours discarded between a fold's training block and its scored block.
# NOT the zero that `crossval.DEFAULT_EMBARGO_HOURS` uses for the
# physics model, and the reason is measured, not assumed: L7.2 put the
# residual's autocorrelation at rho(168) = 0.557 on Fox_education_Claude
# and 0.155 on Bull_education_Luke. Hours either side of a split
# boundary are therefore substantially the same observation, and a
# memorising model scores them by recall. A week is the longest lag L7.2
# reported; note that Claude's ACF had NOT died by then, so even this
# embargo leaves the out-of-fold ML share slightly optimistic. Say so
# when quoting it.
DEFAULT_EMBARGO_HOURS = 168

# Folds. Five gives ~1,300 scored hours per block on a year -- long
# enough that a block spans several weather episodes rather than one.
DEFAULT_N_FOLDS = 5

# Below this many hours a fold's training block cannot support a boosted
# ensemble on six features without memorising outright.
MIN_TRAIN_HOURS = 24 * 14

# When the ML share reaches this multiple of the physics share, the
# artifact has stopped being a physical model with a correction and has
# become a statistical model with a physical prior. That may still be
# useful, but it must not be described as a calibrated physical twin,
# and the warning exists so the description cannot drift silently.
ML_DOMINANCE_RATIO = 1.0

_PERCENT = 100.0
_G_PER_KG = 1000.0
_HOURS_PER_DAY = 24.0

#: A no-argument callable returning an unfitted regressor with
#: scikit-learn's `fit` / `predict` API. Injected rather than hardcoded
#: so a test can substitute something whose predictions are known on
#: paper, and so swapping the learner never touches the decomposition.
ModelFactory = Callable[[], Any]


def default_model_factory(seed: int = SEED) -> ModelFactory:
    """A seeded histogram gradient-boosting regressor, unfitted.

    The import is deferred into the returned closure, following the
    pattern `residual.residual_diagnostics` already uses for
    `scipy.stats.chi2`: importing this module must not require
    scikit-learn, so that everything except the fit itself -- the
    feature builder, the decomposition, the tests for both -- runs in an
    environment that has not installed it.

    Args:
        seed: Passed to the estimator's `random_state`.

    Returns:
        A factory producing a fresh unfitted estimator per call.
    """

    def build() -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            # Shallow by intent. Depth 4 lets the ensemble express
            # "hot AND humid AND occupied", which is the kind of
            # interaction L7.1's marginal profiles could localise but
            # not attribute. Deeper trees start expressing "this
            # specific August afternoon".
            max_depth=4,
            max_leaf_nodes=15,
            # ~2 days of hours in the smallest leaf. A 20-hour leaf is
            # one weather episode, and a correction fitted to one
            # episode is not a correction.
            min_samples_leaf=50,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            # EXPLICITLY OFF. The default, "auto", switches early
            # stopping on once n > 10,000 and then holds out a RANDOM
            # 10% of the rows to decide when to stop -- a random split
            # on an hourly series, which is the exact leak
            # `crossval.neighbour_leak_test` exists to demonstrate. It
            # would not fail loudly; it would quietly stop at the
            # iteration that best interpolates its neighbours. Left off,
            # `max_iter` and the l2 term do the regularising, and the
            # out-of-fold score is the only thing judging the result.
            early_stopping=False,
            random_state=seed,
        )

    return build


def build_features(
    index: pd.DatetimeIndex,
    t_outdoor_c: npt.ArrayLike,
    humidity_ratio_kg_per_kg: npt.ArrayLike,
) -> pd.DataFrame:
    """Assemble the exogenous feature matrix for the residual model.

    The lagged-weather column is built on a COMPLETE hourly grid and
    then read back at the index's own timestamps. On a cleaned BDG2 year
    the index has holes -- M3 drops rows rather than inventing them --
    and a rolling window applied straight to a gappy frame averages "the
    previous 24 ROWS", which after a two-day outage means the previous
    three days. The feature would then be silently wrong at exactly the
    hours around a data-quality event, which is where a reviewer looks
    first.

    Hour of day enters as sine and cosine rather than as an integer.
    A tree can split an integer hour, but only at a threshold: hour 23
    and hour 0 land at opposite ends of every possible split even though
    they are one hour apart. The pair of harmonics makes them adjacent,
    which is the same argument `residual.Binning.CATEGORICAL` makes for
    cyclic drivers.

    Args:
        index: Timestamps of the series, tz-aware or naive, sorted and
            unique.
        t_outdoor_c: Outdoor dry-bulb temperature, degC.
        humidity_ratio_kg_per_kg: Outdoor humidity ratio, kg/kg.

    Returns:
        A frame indexed by `index` with exactly `FEATURE_NAMES` as
        columns, in that order. The lagged-weather column carries NaN
        where fewer than `WEATHER_WINDOW_MIN_PERIODS` hours are
        available; that is a value the estimator handles, not an error.

    Raises:
        ValueError: If the index is not a sorted, unique DatetimeIndex,
            if any array length disagrees with it, or if a weather
            input is non-finite.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"index must be a pd.DatetimeIndex, got {type(index).__name__}")
    if index.size == 0:
        raise ValueError("index is empty")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted; a rolling window on an unsorted index is a lie")
    if index.has_duplicates:
        raise ValueError("index has duplicate timestamps")

    temperature = np.asarray(t_outdoor_c, dtype=float)
    humidity = np.asarray(humidity_ratio_kg_per_kg, dtype=float)
    for name, values in (("t_outdoor_c", temperature), ("humidity_ratio", humidity)):
        if values.shape != (index.size,):
            raise ValueError(
                f"{name} has shape {values.shape}, expected ({index.size},) to match the index"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"{name} contains non-finite values. Cleaning is M3's job; a NaN "
                "arriving here means the pipeline was bypassed."
            )

    # Complete grid -> roll -> read back. `freq="h"` is the project's
    # hourly contract (schema.validate_schema enforces it upstream).
    grid = pd.date_range(index[0], index[-1], freq="h")
    on_grid = pd.Series(temperature, index=index).reindex(grid)
    rolled = on_grid.rolling(
        window=WEATHER_WINDOW_HOURS, min_periods=WEATHER_WINDOW_MIN_PERIODS
    ).mean()

    hour = index.hour.to_numpy(dtype=float)
    angle = 2.0 * np.pi * hour / _HOURS_PER_DAY

    return pd.DataFrame(
        {
            "outdoor_dry_bulb_c": temperature,
            "outdoor_dry_bulb_24h_mean_c": rolled.reindex(index).to_numpy(dtype=float),
            "humidity_ratio_g_per_kg": humidity * _G_PER_KG,
            "hour_sin": np.sin(angle),
            "hour_cos": np.cos(angle),
            "is_weekend": (index.dayofweek.to_numpy() >= 5).astype(float),
        },
        index=index,
        columns=list(FEATURE_NAMES),
    )


@dataclass(frozen=True, eq=False)
class VarianceDecomposition:
    """How much of the measured load each layer explains.

    All three shares are variance shares against the SAME denominator --
    the total sum of squares of the measured load about its own mean.
    That denominator is not arbitrary: it is exactly L6.4's annual-mean
    baseline, so `physics_pct` reads as "how much better than predicting
    the annual mean the calibrated model is", which is the comparison
    the M6 gate was already judged against. A share quoted against any
    other denominator is not comparable to anything else in this project.

    Attributes:
        label: What was decomposed.
        n_hours: Hours entering the decomposition.
        ss_total: Sum of squares of the measured load about its mean.
        ss_physics: Sum of squared physics residuals.
        ss_hybrid: Sum of squared hybrid residuals.
        physics_pct: `100 * (1 - ss_physics / ss_total)`. CAN BE
            NEGATIVE, and that is a real result, not a bug: L6.4's
            uncalibrated physics scored worse than the annual mean on
            this very building.
        ml_pct: `100 * (ss_physics - ss_hybrid) / ss_total`. Can also be
            negative -- a correction that made things worse on hours it
            had not seen.
        unexplained_pct: `100 * ss_hybrid / ss_total`.
    """

    label: str
    n_hours: int
    ss_total: float
    ss_physics: float
    ss_hybrid: float
    physics_pct: float
    ml_pct: float
    unexplained_pct: float

    @property
    def explained_pct(self) -> float:
        """Physics plus ML -- the hybrid's R-squared, as a percentage."""
        return self.physics_pct + self.ml_pct

    @property
    def ml_dominates(self) -> bool:
        """Whether the ML layer explains at least as much as the physics.

        When this is true the deliverable must not be described as a
        calibrated physical model with a correction term. See
        `ML_DOMINANCE_RATIO`.
        """
        return self.ml_pct >= ML_DOMINANCE_RATIO * self.physics_pct

    def summary(self) -> pd.Series:
        """The three shares as a labelled series, for logging."""
        return pd.Series(
            {
                "physics %": self.physics_pct,
                "ML %": self.ml_pct,
                "unexplained %": self.unexplained_pct,
            },
            name=self.label or "decomposition",
        )


def variance_decomposition(
    measured_kw: npt.ArrayLike,
    physics_predicted_kw: npt.ArrayLike,
    hybrid_predicted_kw: npt.ArrayLike,
    *,
    label: str = "",
) -> VarianceDecomposition:
    """Split explained variance into physics, ML and unexplained.

    Args:
        measured_kw: Measured cooling load, kW.
        physics_predicted_kw: The calibrated physical model's prediction.
        hybrid_predicted_kw: Physics plus the ML correction, clipped.
        label: What is being decomposed, for the log.

    Returns:
        The decomposition. The three shares sum to 100.0 exactly, by
        construction rather than by rounding.

    Raises:
        ValueError: If the arrays disagree in shape, hold fewer than two
            points, are non-finite, or if the measured load has zero
            variance (every share would divide by zero, and a constant
            load is a metering fault, not a modelling problem).
    """
    measured = np.asarray(measured_kw, dtype=float)
    physics = np.asarray(physics_predicted_kw, dtype=float)
    hybrid = np.asarray(hybrid_predicted_kw, dtype=float)

    for name, values in (
        ("measured_kw", measured),
        ("physics_predicted_kw", physics),
        ("hybrid_predicted_kw", hybrid),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {values.ndim} dimensions")
        if values.shape != measured.shape:
            raise ValueError(
                f"{name} has {values.size} points, measured_kw has {measured.size}"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    if measured.size < 2:
        raise ValueError("a decomposition needs at least two points")

    ss_total = float(((measured - measured.mean()) ** 2).sum())
    if ss_total == 0.0:
        raise ValueError(
            "measured load has zero variance, so no share is defined. A "
            "perfectly constant meter is a C3 stuck sensor -- see M3."
        )
    ss_physics = float(((measured - physics) ** 2).sum())
    ss_hybrid = float(((measured - hybrid) ** 2).sum())

    decomposition = VarianceDecomposition(
        label=label,
        n_hours=int(measured.size),
        ss_total=ss_total,
        ss_physics=ss_physics,
        ss_hybrid=ss_hybrid,
        physics_pct=_PERCENT * (1.0 - ss_physics / ss_total),
        ml_pct=_PERCENT * (ss_physics - ss_hybrid) / ss_total,
        unexplained_pct=_PERCENT * ss_hybrid / ss_total,
    )

    if decomposition.physics_pct < 0.0:
        logger.warning(
            "%s: physics share is NEGATIVE (%.1f%%) -- the calibrated model is "
            "worse than predicting the annual mean on these hours. Fix the "
            "physics before crediting any ML correction.",
            label or "decomposition",
            decomposition.physics_pct,
        )
    if decomposition.ml_dominates:
        logger.warning(
            "%s: ML share %.1f%% >= physics share %.1f%%. This is no longer a "
            "physical model with a correction term; do not describe it as one.",
            label or "decomposition",
            decomposition.ml_pct,
            decomposition.physics_pct,
        )
    return decomposition


def out_of_fold_correction(
    features: pd.DataFrame,
    residual_kw: npt.ArrayLike,
    folds: tuple[TimeFold, ...],
    *,
    model_factory: ModelFactory | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Predict each fold's residual from a model fitted only on its past.

    The `spin_up` region of every fold is IGNORED here, deliberately.
    Spin-up exists because the 2R2C model carries envelope state that
    has to settle before its predictions mean anything; a tree ensemble
    carries no state at all, so honouring the spin-up would only delete
    training rows for no gain. The `embargo`, which `crossval` defaults
    to zero for the physics model, is what matters instead -- and
    `expanding_window_folds` said so when it was written.

    Args:
        features: Feature matrix, one row per hour, in time order.
        residual_kw: `measured - physics_predicted`, same length.
        folds: Folds from `expanding_window_folds`.
        model_factory: Produces an unfitted estimator. Defaults to
            `default_model_factory()`.

    Returns:
        `(correction_kw, scored_mask)`. `correction_kw` holds the
        out-of-fold prediction where `scored_mask` is True and 0.0
        elsewhere; the mask is False for fold 1's training block and for
        every embargoed hour, which together are roughly a quarter of a
        year. Every downstream number must be computed under this mask.

    Raises:
        ValueError: If shapes disagree, the residual is non-finite, no
            folds were given, or a fold's training block is shorter than
            `MIN_TRAIN_HOURS`.
    """
    residual = np.asarray(residual_kw, dtype=float)
    if residual.ndim != 1 or residual.size != len(features):
        raise ValueError(
            f"residual has {residual.size} points, features have {len(features)} rows"
        )
    if not np.isfinite(residual).all():
        raise ValueError("residual contains non-finite values")
    if not folds:
        raise ValueError("no folds given; see crossval.expanding_window_folds")

    factory = model_factory or default_model_factory()
    matrix = features.to_numpy(dtype=float)
    correction = np.zeros(residual.size, dtype=float)
    scored = np.zeros(residual.size, dtype=bool)

    for fold in folds:
        if fold.n_train < MIN_TRAIN_HOURS:
            raise ValueError(
                f"fold {fold.number} trains on {fold.n_train} hours, below the "
                f"{MIN_TRAIN_HOURS}-hour floor. Use fewer folds."
            )
        model = factory()
        model.fit(matrix[fold.train_slice], residual[fold.train_slice])
        predicted = np.asarray(model.predict(matrix[fold.validate_slice]), dtype=float)
        correction[fold.validate_slice] = predicted
        scored[fold.validate_slice] = True
        logger.info(
            "fold %d: fitted on %d h, corrected %d h (mean correction %+.1f kW)",
            fold.number,
            fold.n_train,
            fold.n_validate,
            float(predicted.mean()),
        )

    return correction, scored


@dataclass(frozen=True, eq=False)
class HybridResult:
    """A physics model, its learnt correction, and what each explained.

    Attributes:
        label: Building and year.
        n_hours_total: Hours in the input series.
        n_hours_scored: Hours the out-of-fold correction covers.
        decomposition: Out-of-fold shares. THE reportable numbers.
        in_sample_decomposition: The same shares from a model fitted on
            and scored over every scored hour. Reported only so the gap
            against `decomposition` can be quoted -- that gap is the
            memorisation, and it is the number a reviewer will ask for.
        physics_cvrmse_pct: CV(RMSE) of the physics model on the scored
            hours.
        hybrid_cvrmse_pct: CV(RMSE) of the hybrid on the same hours.
        physics_nmbe_pct: NMBE of the physics model on those hours.
        hybrid_nmbe_pct: NMBE of the hybrid on those hours.
        clipped_fraction: Share of scored hours where the correction
            drove the hybrid below zero and it was clipped.
        diagnostics_before: L7.2 diagnostics on the physics residual,
            over the scored hours.
        diagnostics_after: The same diagnostics on the hybrid residual,
            over the same hours. If the ML layer took real structure
            out, `daily_variance_share` falls.
        fold_ml_pct: The ML share computed SEPARATELY within each fold's
            scored block, in fold order. Each uses its own block's
            variance as the denominator, so the values are NOT
            comparable to `decomposition.ml_pct` in magnitude -- only
            their signs and their spread are meaningful. Carried because
            the pooled share is a sum of squares over the whole year and
            one large block can carry it entirely: a pooled +3.3% built
            from folds of (+12, +3, +38, +0, -3) is a seasonal
            correction, not a uniform one, and the two are described
            very differently.
        correction_kw: The out-of-fold correction, 0.0 off-mask.
        hybrid_kw: `max(physics + correction, 0)`, full length.
        scored_mask: Which hours the numbers above are computed over.
        folds: The folds used.

    Note:
        `diagnostics_before` and `diagnostics_after` are computed on the
        scored hours CONCATENATED across folds, so the embargo gaps are
        spliced out and a lag straddling a join is mislabelled -- about
        4 joins x 168 lags out of ~6,500 hours at the longest lag. That
        is tolerated for one reason: the splice is IDENTICAL in the
        before and the after series, so it cancels in the comparison,
        and the comparison is the finding. Do not quote either ACF as a
        standalone measurement of the residual -- L7.2's numbers, on the
        unbroken year, are the ones for that.

    Note:
        There is deliberately NO G14 verdict on the hybrid. ASHRAE
        Guideline 14 governs calibrated simulation models with a
        countable parameter set -- its `n - p` correction has no defined
        value for an ensemble of 400 trees, and any number chosen for
        `p` would be an invention dressed as a standard. The CV(RMSE)
        figures here use the physics model's own `p` so the two columns
        are comparable to each other; the project's gate number remains
        L6.10's, on the physics model alone.
    """

    label: str
    n_hours_total: int
    n_hours_scored: int
    decomposition: VarianceDecomposition
    in_sample_decomposition: VarianceDecomposition
    physics_cvrmse_pct: float
    hybrid_cvrmse_pct: float
    physics_nmbe_pct: float
    hybrid_nmbe_pct: float
    clipped_fraction: float
    diagnostics_before: ResidualDiagnostics
    diagnostics_after: ResidualDiagnostics
    fold_ml_pct: tuple[float, ...]
    correction_kw: npt.NDArray[np.float64]
    hybrid_kw: npt.NDArray[np.float64]
    scored_mask: npt.NDArray[np.bool_]
    folds: tuple[TimeFold, ...]

    @property
    def scored_fraction(self) -> float:
        """Share of the input hours the reported numbers cover."""
        return self.n_hours_scored / self.n_hours_total

    @property
    def memorisation_gap_pct(self) -> float:
        """In-sample ML share minus out-of-fold ML share, percentage points.

        Large and positive is the normal, expected result. It becomes a
        finding when it is large enough that the out-of-fold share is
        near zero -- the ensemble learnt the year, not the building.
        """
        return self.in_sample_decomposition.ml_pct - self.decomposition.ml_pct

    @property
    def n_folds_harmed(self) -> int:
        """Folds where the correction made the prediction WORSE.

        The pooled share cannot show this: a large positive block and a
        small negative one add to a positive number. A correction that
        helps in three blocks and hurts in two is a seasonal term, and
        claiming it as a general improvement would be wrong in a way
        that only shows up on a year with a different season mix.
        """
        return sum(1 for share in self.fold_ml_pct if share < 0.0)

    @property
    def cvrmse_improvement_pct(self) -> float:
        """Relative CV(RMSE) reduction from the correction, percent.

        Same definition as `baseline.relative_cvrmse_improvement_pct`,
        so the hybrid's gain over the physics is stated in the units the
        physics model's gain over the baselines was stated in.
        """
        return (
            _PERCENT
            * (self.physics_cvrmse_pct - self.hybrid_cvrmse_pct)
            / self.physics_cvrmse_pct
        )

    def summary(self) -> pd.DataFrame:
        """One row per layer, for the log and the report."""
        return pd.DataFrame(
            {
                "out-of-fold %": [
                    self.decomposition.physics_pct,
                    self.decomposition.ml_pct,
                    self.decomposition.unexplained_pct,
                ],
                "in-sample %": [
                    self.in_sample_decomposition.physics_pct,
                    self.in_sample_decomposition.ml_pct,
                    self.in_sample_decomposition.unexplained_pct,
                ],
            },
            index=["physics", "ML", "unexplained"],
        )


def fit_hybrid(
    index: pd.DatetimeIndex,
    measured_kw: npt.ArrayLike,
    physics_predicted_kw: npt.ArrayLike,
    *,
    t_outdoor_c: npt.ArrayLike,
    humidity_ratio_kg_per_kg: npt.ArrayLike,
    n_physics_params: int,
    label: str = "",
    n_folds: int = DEFAULT_N_FOLDS,
    embargo_hours: int = DEFAULT_EMBARGO_HOURS,
    model_factory: ModelFactory | None = None,
) -> HybridResult:
    """Learn the physics model's residual and decompose what each explains.

    Args:
        index: Hourly timestamps, sorted and unique.
        measured_kw: Measured cooling load, kW.
        physics_predicted_kw: The calibrated 2R2C model's prediction, kW,
            from parameters that are FROZEN. Re-fitting the physics
            after seeing this decomposition and then re-running it makes
            the ML share a function of a choice made with the answer in
            hand.
        t_outdoor_c: Outdoor dry bulb, degC.
        humidity_ratio_kg_per_kg: Outdoor humidity ratio, kg/kg.
        n_physics_params: Calibrated parameter count, for CV(RMSE)'s
            `n - p`. See the note on `HybridResult` about why the hybrid
            reuses it.
        label: Building and year, for logs.
        n_folds: Expanding-window folds.
        embargo_hours: Hours dropped between training and scoring.
        model_factory: Estimator factory; defaults to
            `default_model_factory()`.

    Returns:
        The hybrid result.

    Raises:
        ValueError: If any input is inconsistent, or if the series is
            too short to fold.
    """
    measured = np.asarray(measured_kw, dtype=float)
    physics = np.asarray(physics_predicted_kw, dtype=float)
    if measured.shape != physics.shape:
        raise ValueError(
            f"measured has {measured.size} points, physics has {physics.size}"
        )

    features = build_features(index, t_outdoor_c, humidity_ratio_kg_per_kg)
    if len(features) != measured.size:
        raise ValueError(
            f"index has {len(features)} timestamps, measured has {measured.size} points"
        )

    residual = measured - physics

    # spin_up_hours=0: see out_of_fold_correction's docstring.
    folds = expanding_window_folds(
        measured.size, n_folds=n_folds, spin_up_hours=0, embargo_hours=embargo_hours
    )
    correction, scored = out_of_fold_correction(
        features, residual, folds, model_factory=model_factory
    )

    # A chilled-water load cannot be negative, and a correction is not
    # licensed to break a physical bound the physics model respects --
    # the inverse model already clips at zero. Clipping AFTER the
    # correction rather than constraining the learner keeps the learner
    # simple and makes the violation countable.
    hybrid = np.maximum(physics + correction, 0.0)
    clipped = int(np.count_nonzero((physics + correction < 0.0) & scored))

    decomposition = variance_decomposition(
        measured[scored],
        physics[scored],
        hybrid[scored],
        label=f"{label} out-of-fold".strip(),
    )

    # In-sample twin: same hours, one model, fitted on and scored over
    # all of them. Only ever used for the memorisation gap.
    factory = model_factory or default_model_factory()
    in_sample_model = factory()
    in_sample_model.fit(features.to_numpy(dtype=float)[scored], residual[scored])
    in_sample_hybrid = np.maximum(
        physics[scored]
        + np.asarray(
            in_sample_model.predict(features.to_numpy(dtype=float)[scored]), dtype=float
        ),
        0.0,
    )
    in_sample = variance_decomposition(
        measured[scored],
        physics[scored],
        in_sample_hybrid,
        label=f"{label} in-sample".strip(),
    )

    # Per-fold shares, computed inline rather than through
    # `variance_decomposition` so that five folds do not emit five sets
    # of the same warnings; one summary warning below is more useful
    # than a wall of them.
    fold_shares = []
    for fold in folds:
        block = fold.validate_slice
        ss_block = float(((measured[block] - measured[block].mean()) ** 2).sum())
        if ss_block == 0.0:
            fold_shares.append(float("nan"))
            continue
        ss_p = float(((measured[block] - physics[block]) ** 2).sum())
        ss_h = float(((measured[block] - hybrid[block]) ** 2).sum())
        fold_shares.append(_PERCENT * (ss_p - ss_h) / ss_block)

    before = residual_diagnostics(residual[scored], label=f"{label} physics residual".strip())
    after = residual_diagnostics(
        (measured - hybrid)[scored], label=f"{label} hybrid residual".strip()
    )

    result = HybridResult(
        label=label,
        n_hours_total=int(measured.size),
        n_hours_scored=int(scored.sum()),
        decomposition=decomposition,
        in_sample_decomposition=in_sample,
        physics_cvrmse_pct=cvrmse(measured[scored], physics[scored], n_params=n_physics_params),
        hybrid_cvrmse_pct=cvrmse(measured[scored], hybrid[scored], n_params=n_physics_params),
        physics_nmbe_pct=nmbe(measured[scored], physics[scored], n_params=n_physics_params),
        hybrid_nmbe_pct=nmbe(measured[scored], hybrid[scored], n_params=n_physics_params),
        clipped_fraction=clipped / max(int(scored.sum()), 1),
        diagnostics_before=before,
        diagnostics_after=after,
        fold_ml_pct=tuple(fold_shares),
        correction_kw=correction,
        hybrid_kw=hybrid,
        scored_mask=scored,
        folds=folds,
    )
    logger.info(
        "%s: %d of %d hours scored (%.0f%%); CV(RMSE) %.2f%% -> %.2f%% "
        "(%+.1f%% relative); daily variance share %.3f -> %.3f\n%s",
        label or "hybrid",
        result.n_hours_scored,
        result.n_hours_total,
        _PERCENT * result.scored_fraction,
        result.physics_cvrmse_pct,
        result.hybrid_cvrmse_pct,
        result.cvrmse_improvement_pct,
        before.daily_variance_share,
        after.daily_variance_share,
        result.summary().round(1).to_string(),
    )
    logger.info(
        "%s: ML share within each fold's own block: %s",
        label or "hybrid",
        [round(share, 2) for share in result.fold_ml_pct],
    )
    if result.n_folds_harmed:
        logger.warning(
            "%s: the correction made %d of %d folds WORSE. The pooled ML share "
            "of %.2f%% is carried by the folds where it helps -- describe it as "
            "a seasonal correction, not a general improvement.",
            label or "hybrid",
            result.n_folds_harmed,
            len(folds),
            decomposition.ml_pct,
        )
    if result.clipped_fraction > 0.0:
        logger.warning(
            "%s: the correction drove the hybrid below zero on %.2f%% of scored "
            "hours; those were clipped.",
            label or "hybrid",
            _PERCENT * result.clipped_fraction,
        )
    return result


def permutation_importance_kw(
    features: pd.DataFrame,
    residual_kw: npt.ArrayLike,
    folds: Sequence[TimeFold],
    *,
    seed: int = SEED,
    n_repeats: int = 10,
    model_factory: ModelFactory | None = None,
) -> Mapping[str, float]:
    """Which features the correction actually needs, measured held-out.

    Each fold fits on its own past and is permuted on its own scored
    block -- the same partition `out_of_fold_correction` uses -- and the
    importances are averaged across folds. AVERAGED, not taken from one
    fold, because a single fold's validation window is a single season:
    the last fold of a calendar year scores November and December, and
    permuting outdoor temperature there costs a cooling model almost
    nothing. Reported from that one fold, "temperature does not matter"
    would be a true statement about December and a false one about the
    building.

    An importance here is "how much worse the correction gets on hours
    the model has not seen when this feature is scrambled" -- not "how
    often the trees split on it", which rewards high-cardinality
    features whether or not they help. Held-out permutation is also why
    two correlated features can BOTH read near zero: either one covers
    for the other. That is a real property of the data rather than an
    artifact, and L7.4 is where it gets contrasted with a SHAP plot.

    Args:
        features: Feature matrix.
        residual_kw: The physics residual, kW.
        folds: The folds to fit and permute over.
        seed: Seeds the estimator and the permutations.
        n_repeats: Permutations per feature per fold.
        model_factory: Estimator factory.

    Returns:
        `{feature: mean increase in RMSE, kW}`, descending. A NEGATIVE
        value means scrambling the feature made the correction better on
        unseen hours, i.e. the feature is noise the ensemble had latched
        onto; it is reported rather than clipped to zero, because the
        clipped version hides exactly that.

    Raises:
        ValueError: If shapes disagree or no folds were given.
    """
    from sklearn.inspection import permutation_importance

    residual = np.asarray(residual_kw, dtype=float)
    if residual.size != len(features):
        raise ValueError(
            f"residual has {residual.size} points, features have {len(features)} rows"
        )
    if not folds:
        raise ValueError("no folds given")

    matrix = features.to_numpy(dtype=float)
    totals = np.zeros(len(features.columns), dtype=float)
    for fold in folds:
        model = (model_factory or default_model_factory(seed))()
        model.fit(matrix[fold.train_slice], residual[fold.train_slice])
        # Scored in RMSE, kW -- the same units as the residual, so an
        # importance reads as "scrambling humidity costs 41 kW of
        # accuracy" rather than as a unitless R-squared drop.
        report = permutation_importance(
            model,
            matrix[fold.validate_slice],
            residual[fold.validate_slice],
            scoring="neg_root_mean_squared_error",
            n_repeats=n_repeats,
            random_state=seed,
        )
        totals += np.asarray(report.importances_mean, dtype=float)

    ranked = sorted(
        zip(features.columns, totals / len(folds), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    return MappingProxyType({str(name): float(value) for name, value in ranked})


def _demo() -> None:
    """A synthetic building whose missing terms are known exactly.

    The physics here is incomplete in the two ways this project has
    actually measured. It is linear in outdoor temperature, while the
    truth carries (a) a load that switches on above 28 degC -- the
    hockey stick L7.1c found on Fox_education_Claude, +54.4 kW/K above
    ~16 degC -- and (b) a latent term that only engages above 8 g/kg,
    which is Q11's finding that the ventilation model's fixed supply
    humidity ratio leaves the term inert for most of the year.

    Because the terms are injected, the answer is known before the
    hybrid runs, and the demo is a CHECK rather than an illustration:

        physics       84.9 %   (by construction)
        learnable      9.0 %   (the two switched terms)
        noise          6.1 %   (the injected Gaussian, irreducible)

    A correct hybrid recovers most of the 9.0 out-of-fold and cannot
    touch the 6.1. If the reported ML share is much ABOVE 9, something
    is leaking; far below, the learner or the features are too weak.
    """
    from cooling_twin import set_seed

    rng = set_seed()
    index = pd.date_range("2016-01-01", periods=8760, freq="h")
    hours = np.arange(index.size, dtype=float)

    t_outdoor = 20.0 + 12.0 * np.sin(2.0 * np.pi * (hours - 2000.0) / 8760.0)
    t_outdoor += 5.0 * np.sin(2.0 * np.pi * hours / 24.0)
    humidity = 0.006 + 0.004 * np.sin(2.0 * np.pi * (hours - 2000.0) / 8760.0)

    physics = 400.0 + 30.0 * t_outdoor
    truth = (
        physics
        + 40.0 * np.maximum(t_outdoor - 28.0, 0.0)
        + 25.0 * np.maximum(humidity * _G_PER_KG - 8.0, 0.0)
    )
    measured = truth + rng.normal(0.0, 90.0, size=index.size)

    result = fit_hybrid(
        index,
        measured,
        physics,
        t_outdoor_c=t_outdoor,
        humidity_ratio_kg_per_kg=humidity,
        n_physics_params=5,
        label="synthetic 2016",
    )
    print(f"\nmemorisation gap: {result.memorisation_gap_pct:+.2f} pp")
    print(
        f"CV(RMSE) {result.physics_cvrmse_pct:.2f}% -> {result.hybrid_cvrmse_pct:.2f}%"
        f"  ({result.cvrmse_improvement_pct:+.1f}% relative)"
    )
    print(
        f"daily variance share {result.diagnostics_before.daily_variance_share:.3f}"
        f" -> {result.diagnostics_after.daily_variance_share:.3f}"
        f"  (white noise: {result.diagnostics_after.white_noise_variance_share:.3f})"
    )

    features = build_features(index, t_outdoor, humidity)
    importance = permutation_importance_kw(features, measured - physics, result.folds)
    print("\npermutation importance, kW RMSE (averaged over folds):")
    for name, value in importance.items():
        print(f"  {name:32s} {value:+8.2f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    _demo()

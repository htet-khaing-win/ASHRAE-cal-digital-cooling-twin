"""Naive baselines -- the reference every calibrated result is judged against.

A CV(RMSE) of 22% is not a result. It is a number waiting for a
comparison. If the annual mean alone scores 48%, then 22% is real work;
if the annual mean scores 25%, the entire physics model has bought
almost nothing and something is wrong with either the model or the
building's load being trivially flat.

`06_ASSESSMENT.md`'s M6 supporting requirements make this explicit: the
calibrated model must beat the best baseline by at least 30% relative
CV(RMSE). That check cannot be run until the baselines exist, so they
are built before the optimiser (L6.7), not after it.

Two baselines are provided, in increasing order of how much they know:

    annual mean         knows the target's mean and nothing else
    linear regression   knows outdoor temperature (or any features given)

Both are ordinary least squares fits and both return a `BaselineFit`
that predicts through the same `.predict()` call, so the SAME
`ashrae_g14_pass()` scores the baselines and the calibrated model. A
baseline scored by a different code path than the model it is meant to
be compared against is not a comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_PERCENT = 100.0

# 06_ASSESSMENT.md, M6 supporting requirements: "Calibrated model beats
# best baseline by >= 30% relative CV(RMSE)". Same reasoning as L6.3's
# G14 thresholds -- this is an acceptance criterion, so it lives in code
# where widening it is a visible, deliberate act.
MIN_RELATIVE_IMPROVEMENT_PCT = 30.0


@dataclass(frozen=True)
class BaselineFit:
    """A fitted baseline model.

    Holds the fitted coefficients rather than the predictions, so the
    same fit can be applied to a different period -- fitting on 2016 and
    predicting 2017 is the only honest way to score a baseline against a
    held-out year (L6.10), and a class that stored predictions could not
    do it.

    Attributes:
        name: Human-readable label, used in report tables.
        coefficients: `[intercept, slope_1, ..., slope_k]`. Length is
            `n_params`.
        n_params: Number of fitted parameters `p`, for the `n - p`
            denominator in `metrics.cvrmse`. Counting this correctly is
            what keeps the baseline comparison fair -- see the L6.4
            rationale.
    """

    name: str
    coefficients: npt.NDArray[np.float64]
    n_params: int

    def predict(self, features: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Predict the target for the given feature rows.

        Args:
            features: `(n,)` or `(n, k)` array of predictor values. The
                annual-mean baseline uses only the row count and
                ignores the values -- that is precisely what makes it a
                naive baseline, and keeping the signature uniform is
                what lets a comparison loop score every baseline and
                the calibrated model through one code path.

        Returns:
            Predicted values, shape `(n,)`.

        Raises:
            ValueError: If the number of feature columns does not match
                the columns this baseline was fitted on.
        """
        design = _design_matrix(features, n_slopes=self.n_params - 1)
        return np.asarray(design @ self.coefficients, dtype=float)


def _design_matrix(
    features: npt.ArrayLike,
    n_slopes: int,
) -> npt.NDArray[np.float64]:
    """Build `[1, x_1, ..., x_k]` and check it has the expected width.

    Args:
        features: `(n,)` or `(n, k)` predictor array.
        n_slopes: How many feature columns are expected, i.e. `p - 1`.
            Pass 0 for an intercept-only (annual mean) model.

    Returns:
        Design matrix of shape `(n, n_slopes + 1)`.

    Raises:
        ValueError: If `features` is empty, not 1- or 2-dimensional,
            non-finite, or has a different number of columns than
            `n_slopes`.
    """
    x = np.asarray(features, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"features must be 1- or 2-dimensional, got {x.ndim} dimensions")
    if x.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    if not np.all(np.isfinite(x)):
        raise ValueError(
            "features must be finite -- a NaN outdoor temperature would "
            "propagate silently through lstsq into every coefficient."
        )

    n_rows = x.shape[0]
    if n_slopes == 0:
        return np.ones((n_rows, 1), dtype=float)
    if x.shape[1] != n_slopes:
        raise ValueError(
            f"this baseline was fitted on {n_slopes} feature column(s) but "
            f"was given {x.shape[1]} -- predicting with a different feature "
            "set than the fit used produces a plausible-looking, meaningless "
            "prediction."
        )
    return np.column_stack([np.ones(n_rows, dtype=float), x])


def _fit_ols(
    design: npt.NDArray[np.float64],
    measured: npt.NDArray[np.float64],
    name: str,
) -> BaselineFit:
    """Solve `design @ beta = measured` in the least-squares sense.

    Uses `np.linalg.lstsq` (SVD-based) rather than the normal equations
    `(X'X)^-1 X'y`, which squares the condition number and loses
    precision on collinear features, and rather than adding
    scikit-learn as a dependency for what is one call.

    Args:
        design: `(n, p)` design matrix, intercept column included.
        measured: `(n,)` target values.
        name: Label for the resulting `BaselineFit`.

    Returns:
        The fitted baseline.

    Raises:
        ValueError: If the design matrix is rank-deficient (collinear or
            constant features), which would otherwise yield an
            arbitrary member of a family of equally-good solutions.
    """
    coefficients, _residuals, rank, _singular_values = np.linalg.lstsq(
        design, measured, rcond=None
    )
    n_params = design.shape[1]
    if rank < n_params:
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < {n_params} "
            "columns) -- a constant or collinear feature was passed. "
            "lstsq would return one arbitrary solution out of infinitely "
            "many, and its coefficients would be uninterpretable."
        )
    return BaselineFit(
        name=name,
        coefficients=np.asarray(coefficients, dtype=float),
        n_params=n_params,
    )


def _validated_target(measured: npt.ArrayLike, n_rows: int) -> npt.NDArray[np.float64]:
    """Check the target array and its alignment with the design matrix."""
    y = np.asarray(measured, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"measured must be 1-dimensional, got {y.ndim} dimensions")
    if y.size != n_rows:
        raise ValueError(
            f"measured has {y.size} points but features has {n_rows} rows -- "
            "these must be aligned observations, not merely similar lengths."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError("measured must be finite -- clean NaN in M3, not here")
    return y


def fit_annual_mean(measured: npt.ArrayLike) -> BaselineFit:
    """Fit the weakest defensible baseline: a constant at the mean.

    This is the floor. Any model that cannot beat it has learned
    nothing about the building. Note it fits exactly ONE parameter --
    the mean itself -- so `n_params` is 1, not 0.

    Args:
        measured: Target values `y` from the training period.

    Returns:
        A `BaselineFit` whose `predict()` returns the training mean for
        every row.

    Raises:
        ValueError: If `measured` is empty, non-1-D, or non-finite.
    """
    y = np.asarray(measured, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"measured must be 1-dimensional, got {y.ndim} dimensions")
    if y.size == 0:
        raise ValueError("measured must contain at least one point")
    if not np.all(np.isfinite(y)):
        raise ValueError("measured must be finite -- clean NaN in M3, not here")

    design = np.ones((y.size, 1), dtype=float)
    return _fit_ols(design, y, name="annual mean")


def fit_linear_regression(
    features: npt.ArrayLike,
    measured: npt.ArrayLike,
    name: str = "linear regression",
) -> BaselineFit:
    """Fit an ordinary least squares baseline on the given features.

    With outdoor dry-bulb temperature as the single feature this is the
    classic utility-analysis regression: cooling load rises roughly
    linearly with outdoor temperature above a balance point. It knows
    nothing about thermal mass, occupancy, or humidity -- which is
    exactly what the RC model is supposed to add, and therefore exactly
    what the comparison is measuring.

    Args:
        features: `(n,)` or `(n, k)` predictor array from the training
            period, e.g. outdoor dry-bulb temperature.
        measured: `(n,)` target values, aligned with `features`.
        name: Label for the resulting `BaselineFit`.

    Returns:
        The fitted baseline, with `n_params = k + 1` (slopes plus
        intercept).

    Raises:
        ValueError: If the arrays are misaligned, empty, non-finite, or
            if the features are collinear or constant.
    """
    x = np.asarray(features, dtype=float)
    n_slopes = 1 if x.ndim == 1 else x.shape[1] if x.ndim == 2 else 0
    if x.ndim == 2 and n_slopes == 0:
        raise ValueError(
            "features has zero columns -- fitting no slopes would silently "
            "produce an annual-mean baseline under a regression's name. "
            "Call fit_annual_mean() if that is what you want."
        )
    design = _design_matrix(features, n_slopes=n_slopes)
    y = _validated_target(measured, n_rows=design.shape[0])
    return _fit_ols(design, y, name=name)


def relative_cvrmse_improvement_pct(
    baseline_cvrmse_pct: float,
    model_cvrmse_pct: float,
) -> float:
    """How much better the model is than the baseline, in relative percent.

    `(baseline - model) / baseline * 100`. Relative, not absolute:
    going from 48% to 34% (a 29% relative gain) and from 10% to 7% (a
    30% relative gain) are comparable amounts of work, while their
    absolute gaps -- 14 points versus 3 -- are not.

    Args:
        baseline_cvrmse_pct: The baseline's CV(RMSE), percent.
        model_cvrmse_pct: The candidate model's CV(RMSE), percent.

    Returns:
        Relative improvement in percent. Negative if the model is worse
        than the baseline.

    Raises:
        ValueError: If `baseline_cvrmse_pct` is not strictly positive.
    """
    if baseline_cvrmse_pct <= 0.0:
        raise ValueError(
            f"baseline_cvrmse_pct must be > 0, got {baseline_cvrmse_pct}. "
            "A baseline with zero error means the target is constant or the "
            "baseline saw the answer -- either way the comparison is void."
        )
    return (baseline_cvrmse_pct - model_cvrmse_pct) / baseline_cvrmse_pct * _PERCENT


def beats_baseline(
    baseline_cvrmse_pct: float,
    model_cvrmse_pct: float,
    min_improvement_pct: float = MIN_RELATIVE_IMPROVEMENT_PCT,
) -> bool:
    """Apply 06_ASSESSMENT.md's >= 30% relative CV(RMSE) requirement.

    Args:
        baseline_cvrmse_pct: The BEST baseline's CV(RMSE), percent --
            not the weakest. Beating only the annual mean is not the
            requirement.
        model_cvrmse_pct: The calibrated model's CV(RMSE), percent.
        min_improvement_pct: Required relative improvement. Defaults to
            the project's 30%.

    Returns:
        True if the model clears the bar. Inclusive at the boundary.

    Raises:
        ValueError: See `relative_cvrmse_improvement_pct`.
    """
    improvement = relative_cvrmse_improvement_pct(baseline_cvrmse_pct, model_cvrmse_pct)
    return improvement >= min_improvement_pct


if __name__ == "__main__":
    from cooling_twin import set_seed
    from cooling_twin.calibration.metrics import cvrmse

    logging.basicConfig(level=logging.INFO)

    # A synthetic year whose load depends on outdoor temperature plus a
    # daily occupancy cycle the temperature alone cannot explain. That
    # residual structure is what a physics model is supposed to capture
    # and a temperature regression is not.
    rng = set_seed()
    hours = np.arange(8760)
    outdoor_c = 15.0 + 12.0 * np.sin(2 * np.pi * (hours - 2160) / 8760)
    occupancy = np.where((hours % 24 >= 8) & (hours % 24 < 18), 1.0, 0.35)
    load_kw = 400.0 + 60.0 * outdoor_c + 500.0 * occupancy + rng.normal(0, 80, 8760)

    mean_fit = fit_annual_mean(load_kw)
    regression_fit = fit_linear_regression(outdoor_c, load_kw)

    logger.info("--- baselines fitted on a synthetic year (n=8760) ---")
    for fit in (mean_fit, regression_fit):
        score = cvrmse(load_kw, fit.predict(outdoor_c), n_params=fit.n_params)
        logger.info(
            "%-20s p=%d  coefficients=%s  CV(RMSE)=%.2f%%",
            fit.name,
            fit.n_params,
            np.round(fit.coefficients, 2),
            score,
        )

    best_baseline = min(
        cvrmse(load_kw, f.predict(outdoor_c), n_params=f.n_params)
        for f in (mean_fit, regression_fit)
    )

    logger.info("--- what a calibrated model would have to reach ---")
    logger.info(
        "best baseline CV(RMSE) = %.2f%%  ->  model must be <= %.2f%%",
        best_baseline,
        best_baseline * (1 - MIN_RELATIVE_IMPROVEMENT_PCT / _PERCENT),
    )
    for candidate in (best_baseline * 0.95, best_baseline * 0.70, best_baseline * 0.50):
        logger.info(
            "candidate CV(RMSE)=%6.2f%%  improvement=%+6.2f%%  beats baseline: %s",
            candidate,
            relative_cvrmse_improvement_pct(best_baseline, candidate),
            beats_baseline(best_baseline, candidate),
        )

"""ASHRAE Guideline 14 calibration metrics -- NMBE and CV(RMSE).

These two numbers are this project's headline result. Everything M6
builds -- baselines (L6.4), the objective function (L6.6), the
optimiser (L6.7), the test-set opening (L6.10) -- is ultimately scored
by the two functions in this file, so they are the last place in the
repo where an off-by-one or a sign convention may be wrong.

Both formulas are transcribed from `03_DOMAIN_REFERENCE.md` SS4:

    NMBE    =  sum(y_i - yhat_i) / ((n - p) * ybar) * 100%

    CVRMSE  =  sqrt( sum((y_i - yhat_i)^2) / (n - p) ) / ybar * 100%

where `n` is the number of data points, `p` the number of calibrated
parameters, and `ybar` the mean of the MEASURED data.

Sign convention (`y - yhat`, measured minus predicted) is the
standard's, and is kept here even though it is the opposite of the
intuitive "error = prediction - truth":

    NMBE > 0  ->  the model UNDER-predicts (measured exceeds predicted)
    NMBE < 0  ->  the model OVER-predicts

L6.1's exploratory notebook reported its bias the other way round
(predicted - measured = -90.7%); the same fit scored through
`nmbe()` reads +90.7%. Anything comparing the two must flip one of
them. See the "Why it's written this way" note on this in the lesson.
"""

from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_PERCENT = 100.0


def _validated_errors(
    measured: npt.ArrayLike,
    predicted: npt.ArrayLike,
    n_params: int,
) -> tuple[npt.NDArray[np.float64], int, float]:
    """Validate a metric's inputs and return the shared intermediates.

    Both `nmbe()` and `cvrmse()` need exactly the same three things --
    the error vector, the degrees of freedom `n - p`, and `ybar` -- and
    exactly the same five failure modes. Computing them in one place
    means the two metrics can never disagree about what counts as a
    valid input, which is the failure a reviewer would actually look
    for: a model that passes NMBE's validation but silently returns
    `nan` from CV(RMSE) on the same data.

    Args:
        measured: Measured values `y`, e.g. metered cooling load.
        predicted: Model output `yhat`, aligned element-wise with
            `measured`.
        n_params: Number of parameters `p` adjusted during calibration.
            Pass 0 for an uncalibrated model (L6.1) or a baseline that
            fits nothing (L6.4's annual mean fits one, its mean).

    Returns:
        `(errors, dof, y_bar)` where `errors` is `measured - predicted`
        (the standard's sign convention), `dof` is `n - p`, and
        `y_bar` is the mean of `measured`.

    Raises:
        ValueError: If the arrays have different lengths, are empty,
            contain non-finite values, if `n_params` is negative or
            leaves no degrees of freedom (`n - p <= 0`), or if the
            measured mean is zero (the metrics are normalised by it and
            are undefined there).
    """
    y = np.asarray(measured, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)

    if y.shape != y_hat.shape:
        raise ValueError(
            f"measured {y.shape} and predicted {y_hat.shape} must have the "
            "same shape -- these metrics compare paired observations, so a "
            "length mismatch means the series are misaligned, not merely "
            "different sizes."
        )
    if y.size == 0:
        raise ValueError("measured and predicted must contain at least one point")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(y_hat))):
        raise ValueError(
            "measured and predicted must be finite -- drop or impute NaN "
            "before scoring (M3's cleaning pipeline is where that belongs). "
            "Silently ignoring NaN here would change n without changing p."
        )
    if n_params < 0:
        raise ValueError(f"n_params must be >= 0, got {n_params}")

    n = int(y.size)
    dof = n - n_params
    if dof <= 0:
        raise ValueError(
            f"n - p must be > 0, got n={n} and p={n_params}. A model with at "
            "least as many parameters as data points can fit the data exactly "
            "and its error metrics carry no information."
        )

    y_bar = float(np.mean(y))
    if y_bar == 0.0:
        raise ValueError(
            "mean of measured is zero -- NMBE and CV(RMSE) are normalised by "
            "it and are undefined. This usually means the wrong column was "
            "passed, or a fully-zero meter period reached the metric."
        )

    return y - y_hat, dof, y_bar


def nmbe(
    measured: npt.ArrayLike,
    predicted: npt.ArrayLike,
    n_params: int,
) -> float:
    """Normalised Mean Bias Error, in percent (ASHRAE G14).

    Measures whether the model is the right SIZE. Positive and negative
    errors cancel, by design -- that is what makes it a *bias* metric,
    and also why `03_DOMAIN_REFERENCE.md` SS4 lists "reporting NMBE
    alone" as a review-catchable mistake. Always report it alongside
    `cvrmse()`.

    Args:
        measured: Measured values `y`.
        predicted: Model output `yhat`, aligned with `measured`.
        n_params: Number of calibrated parameters `p`.

    Returns:
        NMBE in percent. Positive means the model under-predicts.

    Raises:
        ValueError: See `_validated_errors`.
    """
    errors, dof, y_bar = _validated_errors(measured, predicted, n_params)
    return float(np.sum(errors) / (dof * y_bar) * _PERCENT)


def cvrmse(
    measured: npt.ArrayLike,
    predicted: npt.ArrayLike,
    n_params: int,
) -> float:
    """Coefficient of Variation of the RMSE, in percent (ASHRAE G14).

    Measures whether the model has the right SHAPE as well as the right
    size. Squaring before summing means positive and negative errors
    reinforce instead of cancelling, so a model that is right on average
    but wrong every single hour scores badly here while scoring ~0 on
    `nmbe()`.

    Args:
        measured: Measured values `y`.
        predicted: Model output `yhat`, aligned with `measured`.
        n_params: Number of calibrated parameters `p`.

    Returns:
        CV(RMSE) in percent. Always >= 0.

    Raises:
        ValueError: See `_validated_errors`.
    """
    errors, dof, y_bar = _validated_errors(measured, predicted, n_params)
    root_mean_square_error = float(np.sqrt(np.sum(errors**2) / dof))
    return root_mean_square_error / y_bar * _PERCENT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. Known-answer case, small enough to check by hand.
    measured_demo = [100.0, 200.0, 300.0, 400.0]
    predicted_demo = [110.0, 190.0, 310.0, 390.0]
    # errors (y - yhat) = [-10, +10, -10, +10] -> they cancel exactly.
    logger.info("--- 1. errors that cancel (n=4, p=0) ---")
    logger.info("NMBE     = %+.4f %%", nmbe(measured_demo, predicted_demo, n_params=0))
    logger.info("CV(RMSE) = %.4f %%", cvrmse(measured_demo, predicted_demo, n_params=0))

    # 2. The same data, but the model is a flat line at the measured mean.
    #    This is L6.4's naive baseline, previewed: NMBE is exactly 0 and the
    #    model contains no information at all.
    mean_model = [250.0] * 4
    logger.info("--- 2. flat model at the measured mean (the NMBE trap) ---")
    logger.info("NMBE     = %+.4f %%", nmbe(measured_demo, mean_model, n_params=0))
    logger.info("CV(RMSE) = %.4f %%", cvrmse(measured_demo, mean_model, n_params=0))

    # 3. What p actually costs. Identical errors, more claimed parameters --
    #    a uniformly 25-low model, so NMBE is non-zero and moves too.
    biased_model = [75.0, 175.0, 275.0, 375.0]
    logger.info("--- 3. the n - p correction (same errors, rising p) ---")
    for p in (0, 1, 2, 3):
        logger.info(
            "p=%d  ->  NMBE %+8.4f %%   CV(RMSE) %8.4f %%",
            p,
            nmbe(measured_demo, biased_model, n_params=p),
            cvrmse(measured_demo, biased_model, n_params=p),
        )

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
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_PERCENT = 100.0


class DataInterval(Enum):
    """The averaging interval the metrics were computed at.

    G14 sets different thresholds for hourly and monthly data, and the
    interval is not inferable from the arrays themselves -- 8,760
    numbers could be hourly for a year or monthly for 730 years. It has
    to be stated, so it is an argument, and an enum rather than a
    string so a typo is a NameError at import rather than a silently
    wrong verdict at report time.
    """

    HOURLY = "hourly"
    MONTHLY = "monthly"


# 03_DOMAIN_REFERENCE.md SS4, "Acceptance thresholds". These are the
# published values of a consensus standard, NOT project settings --
# deliberately module constants rather than entries in
# config/calibration.yaml. See the L6.3 rationale: a threshold that
# lives in a config file is a threshold that can be widened by whoever
# is failing it, which destroys the only property that makes an
# external standard worth using.
_THRESHOLDS: MappingProxyType[DataInterval, tuple[float, float]] = MappingProxyType(
    {
        # interval: (|NMBE| limit %, CV(RMSE) limit %)
        DataInterval.HOURLY: (10.0, 30.0),
        DataInterval.MONTHLY: (5.0, 15.0),
    }
)

# 03_DOMAIN_REFERENCE.md SS4: "Stretch target: CV(RMSE) <= 20%."
_STRETCH_CVRMSE_PCT = 20.0

# 06_ASSESSMENT.md, "On suspiciously good results": hourly building
# energy prediction below 5% CV(RMSE) is implausible and should be
# investigated, not celebrated.
_SUSPICIOUSLY_GOOD_CVRMSE_PCT = 5.0


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


@dataclass(frozen=True)
class G14Verdict:
    """The full result of an ASHRAE Guideline 14 acceptance check.

    Deliberately not a bare `bool`. A bool answers "did it pass" and
    throws away every question actually asked next: which of the two
    criteria failed, by how much, against which interval's thresholds,
    and what the numbers were. All of those are needed for
    `reports/02_calibration.md`'s required table (06_ASSESSMENT.md M6)
    and for L6.10's one-time test-set log entry, and reconstructing
    them by re-running the metrics alongside the bool invites the two
    call sites drifting apart.

    Attributes:
        interval: Which threshold set was applied.
        nmbe_pct: Measured NMBE, percent (sign convention as `nmbe()`).
        cvrmse_pct: Measured CV(RMSE), percent.
        nmbe_limit_pct: The |NMBE| limit applied, percent.
        cvrmse_limit_pct: The CV(RMSE) limit applied, percent.
        n_params: Number of calibrated parameters used in `n - p`.
        n_points: Number of paired observations scored.
    """

    interval: DataInterval
    nmbe_pct: float
    cvrmse_pct: float
    nmbe_limit_pct: float
    cvrmse_limit_pct: float
    n_params: int
    n_points: int

    @property
    def nmbe_pass(self) -> bool:
        """|NMBE| <= the interval's limit. Note: inclusive."""
        return abs(self.nmbe_pct) <= self.nmbe_limit_pct

    @property
    def cvrmse_pass(self) -> bool:
        """CV(RMSE) <= the interval's limit. Note: inclusive."""
        return self.cvrmse_pct <= self.cvrmse_limit_pct

    @property
    def passed(self) -> bool:
        """G14 requires BOTH criteria. Neither alone is sufficient."""
        return self.nmbe_pass and self.cvrmse_pass

    @property
    def meets_stretch_target(self) -> bool:
        """CV(RMSE) <= 20% -- this project's own target, not G14's."""
        return self.cvrmse_pct <= _STRETCH_CVRMSE_PCT

    @property
    def is_suspiciously_good(self) -> bool:
        """CV(RMSE) < 5% on hourly data (06_ASSESSMENT.md).

        A true condition here is NOT a pass signal. It is a prompt to
        check for a leak, a trivially-predictable target, or the test
        set having been touched.
        """
        return (
            self.interval is DataInterval.HOURLY
            and self.cvrmse_pct < _SUSPICIOUSLY_GOOD_CVRMSE_PCT
        )

    def summary(self) -> str:
        """One-line human-readable verdict, for logs and reports."""
        return (
            f"{'PASS' if self.passed else 'FAIL'} "
            f"({self.interval.value}, n={self.n_points}, p={self.n_params}): "
            f"NMBE {self.nmbe_pct:+.2f}% "
            f"[{'ok' if self.nmbe_pass else 'FAIL'}, limit +/-{self.nmbe_limit_pct:g}%], "
            f"CV(RMSE) {self.cvrmse_pct:.2f}% "
            f"[{'ok' if self.cvrmse_pass else 'FAIL'}, limit {self.cvrmse_limit_pct:g}%]"
        )


def ashrae_g14_pass(
    measured: npt.ArrayLike,
    predicted: npt.ArrayLike,
    n_params: int,
    interval: DataInterval = DataInterval.HOURLY,
) -> G14Verdict:
    """Score a model against ASHRAE Guideline 14's acceptance criteria.

    Computes both metrics and applies the interval's published
    thresholds. A model passes only if BOTH criteria pass.

    This function does not know, and must not know, whether it is being
    handed train or test data. That distinction is a matter of process
    (ADR-002's locked split, and L6.10's one-time opening), enforced by
    the caller and logged in `07_PROGRESS.md` -- not something a
    scoring function can police.

    Args:
        measured: Measured values `y`.
        predicted: Model output `yhat`, aligned with `measured`.
        n_params: Number of calibrated parameters `p`.
        interval: Averaging interval of the data. Defaults to hourly,
            this project's declared target (03_DOMAIN_REFERENCE.md SS4).

    Returns:
        A `G14Verdict` carrying both metrics, both thresholds, and the
        per-criterion outcomes.

    Raises:
        ValueError: If `interval` is not a `DataInterval`, or for any
            of the input problems listed in `_validated_errors`.
    """
    if interval not in _THRESHOLDS:
        raise ValueError(
            f"interval must be a DataInterval, got {interval!r}. "
            "G14 publishes thresholds only for hourly and monthly data; "
            "any other interval needs its own justification, not a "
            "borrowed threshold."
        )

    nmbe_limit_pct, cvrmse_limit_pct = _THRESHOLDS[interval]
    verdict = G14Verdict(
        interval=interval,
        nmbe_pct=nmbe(measured, predicted, n_params),
        cvrmse_pct=cvrmse(measured, predicted, n_params),
        nmbe_limit_pct=nmbe_limit_pct,
        cvrmse_limit_pct=cvrmse_limit_pct,
        n_params=n_params,
        n_points=int(np.asarray(measured, dtype=float).size),
    )

    if verdict.is_suspiciously_good:
        logger.warning(
            "CV(RMSE) of %.2f%% is below the %.1f%% plausibility floor for "
            "hourly building energy prediction (06_ASSESSMENT.md). Check for "
            "a data leak, a trivially-predictable target, or the test set "
            "having been touched, before reporting this as a result.",
            verdict.cvrmse_pct,
            _SUSPICIOUSLY_GOOD_CVRMSE_PCT,
        )

    return verdict


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

    # 4. The G14 gate, on a synthetic year. set_seed() keeps this
    #    reproducible (L0.3) -- a demo whose verdict changes between runs
    #    is worse than no demo.
    from cooling_twin import set_seed

    rng = set_seed()
    hours = np.arange(8760)
    truth = 2000.0 + 600.0 * np.sin(2 * np.pi * hours / 8760) + rng.normal(0, 150, 8760)

    logger.info("--- 4. G14 gate, hourly, n=8760, p=8 ---")
    for label, model in (
        ("uncalibrated (90% low)", truth * 0.1),
        ("biased but well-shaped", truth * 0.85),
        ("plausibly calibrated", truth + rng.normal(0, 300, 8760)),
    ):
        verdict = ashrae_g14_pass(truth, model, n_params=8)
        logger.info("%-24s %s", label, verdict.summary())

    # 5. Same numbers, monthly thresholds -- the standard is stricter when
    #    averaging has already removed most of the hour-to-hour variance.
    logger.info("--- 5. hourly vs monthly thresholds on identical data ---")
    borderline = truth * 0.93
    for data_interval in DataInterval:
        logger.info(
            "%-8s %s",
            data_interval.value,
            ashrae_g14_pass(truth, borderline, n_params=8, interval=data_interval).summary(),
        )

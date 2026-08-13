"""Cross-validation for a time series -- and why the obvious kind is a lie.

Everything M6 has measured so far is a TRAINING number. CV(RMSE) 11.72%
on Fox_education_Claude says the calibrated model reproduces the year it
was fitted to. It does not say the model would reproduce a year it has
not seen, and those two claims are separated by exactly one thing:
whether the parameters captured the building's physics or the year's
particular weather.

The held-out year (2017) answers that, and it is opened ONCE, at L6.10,
deliberately (ADR-002). This module is how the question gets asked
without spending it: split the TRAINING year into blocks, fit on the
earlier ones, score on the later ones, and look at the gap.

    fold 1   |TTTTTT|~~|VVVV|........................|
    fold 2   |TTTTTTTTTT|~~|VVVV|....................|
    fold 3   |TTTTTTTTTTTTTT|~~|VVVV|................|
             ^ train        ^ spin-up (discarded)
                               ^ validated against measurements

WHY NOT `sklearn.model_selection.KFold`, or any random split. Because
it does not test anything on a series like this one. Cooling load at
14:00 and cooling load at 15:00 are nearly the same number: hold out a
random 20% of hours and every held-out hour still has both of its
neighbours in the training set, so a model does not need to know any
physics to score well -- it only needs to interpolate. `neighbour_leak_test`
in this module measures exactly that, and on a real building it returns
a CV(RMSE) that would pass ASHRAE G14 comfortably while knowing nothing
about the building at all. A split that a two-line interpolator can beat
is not a test.

WHY NOT `sklearn.model_selection.TimeSeriesSplit`, which does block
correctly. Two reasons, in order of importance. This project's model is
a state-carrying ODE: a validation block starting in July cannot be
simulated from nowhere, because the envelope temperature at that instant
depends on the weeks before it. The simulation has to start EARLIER than
the scoring window and the spin-up predictions have to be thrown away --
a concept `TimeSeriesSplit` has no way to express, since it yields index
arrays rather than contiguous windows. And scikit-learn is not a
dependency of this project (L6.4 declined it for `lstsq`); adding it for
one splitter that would then need wrapping anyway is the wrong trade.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cooling_twin import SEED
from cooling_twin.calibration.metrics import DataInterval, cvrmse, g14_thresholds, nmbe

logger = logging.getLogger(__name__)

# Hours of simulation run BEFORE a validation window whose predictions
# are discarded. The 2R2C envelope node's slow mode was measured at
# ~4.9 h in L4.3, so 72 h is about fifteen time constants: any error in
# the assumed initial temperature has decayed by e^-15 (~3e-7) before
# the first scored hour. Shorter would score the guess about the initial
# condition as if it were model error; much longer only costs data.
DEFAULT_SPIN_UP_HOURS = 72

# Hours discarded BETWEEN the end of training and the start of
# validation. Zero by default, and that is a considered position rather
# than an omission -- see `expanding_window_folds`.
DEFAULT_EMBARGO_HOURS = 0

# A validation window shorter than this is not scored. Two weeks is the
# shortest window that contains both a weekend and a weather swing on
# this dataset; below that, one hot spell decides the fold.
MIN_VALIDATE_HOURS = 336

# A gap this large between training and validation CV(RMSE), in
# PERCENTAGE POINTS, is reported as an overfitting signal. Not a
# statistical test -- there is no null distribution here -- but a
# threshold that has to be stated in advance, because a threshold chosen
# after seeing the folds is not a threshold.
OVERFITTING_GAP_PCT = 5.0


@dataclass(frozen=True)
class TimeFold:
    """One contiguous train / spin-up / validate partition of a series.

    Positional bounds rather than index arrays, because the model is a
    simulation: it needs an unbroken stretch of hours, and an index
    array cannot promise one.

    Attributes:
        number: 1-based fold number, for logs and reports.
        train_start: First training sample (inclusive).
        train_stop: One past the last training sample.
        spin_up_start: Where the validation SIMULATION begins. Earlier
            than `validate_start`; its predictions are discarded.
        validate_start: First scored sample (inclusive).
        validate_stop: One past the last scored sample.
    """

    number: int
    train_start: int
    train_stop: int
    spin_up_start: int
    validate_start: int
    validate_stop: int

    @property
    def n_train(self) -> int:
        """Training samples in this fold."""
        return self.train_stop - self.train_start

    @property
    def n_validate(self) -> int:
        """Scored samples in this fold."""
        return self.validate_stop - self.validate_start

    @property
    def n_spin_up(self) -> int:
        """Simulated-but-discarded samples before the scored window."""
        return self.validate_start - self.spin_up_start

    @property
    def train_slice(self) -> slice:
        """Slice selecting the training samples."""
        return slice(self.train_start, self.train_stop)

    @property
    def simulate_slice(self) -> slice:
        """Slice selecting spin-up AND validation -- what to simulate."""
        return slice(self.spin_up_start, self.validate_stop)

    @property
    def validate_slice(self) -> slice:
        """Slice selecting the scored samples."""
        return slice(self.validate_start, self.validate_stop)

    @property
    def scored_offset(self) -> int:
        """Where the scored window starts inside `simulate_slice`.

        The one index that is easy to get wrong: a simulation run over
        `simulate_slice` must be sliced from here before scoring, or the
        spin-up hours are scored as if they were predictions.
        """
        return self.validate_start - self.spin_up_start


@dataclass(frozen=True)
class FoldScore:
    """What one fold measured.

    Attributes:
        fold: The partition scored.
        train_cvrmse_pct: CV(RMSE) on the fold's own training window.
        validate_cvrmse_pct: CV(RMSE) on the held-out window.
        train_nmbe_pct: NMBE on the training window.
        validate_nmbe_pct: NMBE on the held-out window.
        parameters: The parameters fitted on this fold's training data.
    """

    fold: TimeFold
    train_cvrmse_pct: float
    validate_cvrmse_pct: float
    train_nmbe_pct: float
    validate_nmbe_pct: float
    parameters: dict[str, float]

    def passes_g14(self, interval: DataInterval = DataInterval.HOURLY) -> bool:
        """Whether the HELD-OUT window meets both G14 criteria.

        Both, not just CV(RMSE). A fold can sit comfortably inside the
        30% scatter limit and still miss the +-10% bias limit, and
        reporting the fold on its CV(RMSE) alone would call that a pass
        -- which is L6.2's NMBE trap wearing a cross-validation costume.
        Thresholds come from `g14_thresholds` rather than being restated
        here, so this cannot drift from `ashrae_g14_pass`.

        Args:
            interval: Which G14 threshold set applies.

        Returns:
            True only if both criteria are met on the validation window.
        """
        nmbe_limit_pct, cvrmse_limit_pct = g14_thresholds(interval)
        return (
            abs(self.validate_nmbe_pct) <= nmbe_limit_pct
            and self.validate_cvrmse_pct <= cvrmse_limit_pct
        )

    @property
    def gap_pct(self) -> float:
        """Validation CV(RMSE) minus training CV(RMSE), percentage points.

        The number this whole module exists to produce. Near zero means
        the fit generalises within the year. Large and positive means it
        memorised. Large and NEGATIVE means the validation window is
        easier than the training window -- a statement about the
        seasons, not about the model, and the reason folds are reported
        individually rather than only as a mean.
        """
        return self.validate_cvrmse_pct - self.train_cvrmse_pct


@dataclass(frozen=True)
class CrossValidationResult:
    """Every fold, plus the summary that must not be read without them.

    Attributes:
        scores: One `FoldScore` per fold, in time order.
        parameter_names: Calibrated parameter names.
        overfitting_gap_pct: The threshold the verdict used.
    """

    scores: tuple[FoldScore, ...]
    parameter_names: tuple[str, ...]
    overfitting_gap_pct: float = OVERFITTING_GAP_PCT

    @property
    def mean_validate_cvrmse_pct(self) -> float:
        """Mean held-out CV(RMSE) across folds."""
        return float(np.mean([score.validate_cvrmse_pct for score in self.scores]))

    @property
    def worst_validate_cvrmse_pct(self) -> float:
        """Worst held-out CV(RMSE) -- the number a reviewer will ask for."""
        return float(np.max([score.validate_cvrmse_pct for score in self.scores]))

    @property
    def mean_gap_pct(self) -> float:
        """Mean generalisation gap, percentage points."""
        return float(np.mean([score.gap_pct for score in self.scores]))

    @property
    def overfitting_suspected(self) -> bool:
        """Whether the mean gap exceeds the declared threshold."""
        return self.mean_gap_pct > self.overfitting_gap_pct

    @property
    def folds_failing_g14(self) -> tuple[int, ...]:
        """Fold numbers whose HELD-OUT window fails G14 on either criterion.

        Reported separately from the gap, because the two say different
        things. A small gap means the model generalises as well as it
        fits; it does not mean either number is acceptable. A fold can
        generalise perfectly and fail the standard on both windows.
        """
        return tuple(
            score.fold.number for score in self.scores if not score.passes_g14()
        )

    def parameter_stability(self) -> dict[str, tuple[float, float]]:
        """`{name: (min, max)}` for each parameter across the folds.

        A parameter whose fitted value swings between folds is not being
        measured by the data -- it is absorbing whatever that stretch of
        the year happened to look like. This is L6.8's identifiability
        question asked a second way, and the two answers should agree;
        where they disagree, the folds are the stronger evidence,
        because each fold is an independent fit rather than a
        re-optimisation of the same one.
        """
        return {
            name: (
                min(score.parameters[name] for score in self.scores),
                max(score.parameters[name] for score in self.scores),
            )
            for name in self.parameter_names
        }

    def summary(self) -> str:
        """Multi-line human-readable summary, for logs and reports."""
        lines = [
            f"{'fold':<6}{'train h':>9}{'val h':>7}"
            f"{'train CV%':>11}{'val CV%':>10}{'gap pp':>9}"
            f"{'val NMBE%':>11}{'G14':>7}",
        ]
        for score in self.scores:
            lines.append(
                f"{score.fold.number:<6}{score.fold.n_train:>9}"
                f"{score.fold.n_validate:>7}{score.train_cvrmse_pct:>11.2f}"
                f"{score.validate_cvrmse_pct:>10.2f}{score.gap_pct:>+9.2f}"
                f"{score.validate_nmbe_pct:>+11.2f}"
                f"{'PASS' if score.passes_g14() else 'FAIL':>7}"
            )
        if self.folds_failing_g14:
            lines.append(
                "  G14 FAILED on held-out fold(s): "
                + ", ".join(str(number) for number in self.folds_failing_g14)
                + " -- a small mean gap does not make a fold acceptable"
            )
        lines.append(
            f"mean validation CV(RMSE) {self.mean_validate_cvrmse_pct:.2f}%, "
            f"worst {self.worst_validate_cvrmse_pct:.2f}%, "
            f"mean gap {self.mean_gap_pct:+.2f} pp "
            f"({'OVERFITTING SUSPECTED' if self.overfitting_suspected else 'within threshold'} "
            f"at {self.overfitting_gap_pct:.1f} pp)"
        )
        return "\n".join(lines)


def expanding_window_folds(
    n_samples: int,
    n_folds: int = 4,
    spin_up_hours: int = DEFAULT_SPIN_UP_HOURS,
    embargo_hours: int = DEFAULT_EMBARGO_HOURS,
    min_validate_hours: int = MIN_VALIDATE_HOURS,
) -> tuple[TimeFold, ...]:
    """Split a series into expanding-window train/validate folds.

    Training always starts at sample 0 and grows; validation is the
    block immediately after it. Every fold trains only on the PAST of
    what it scores, which is the whole point -- a building model that
    needs next August to predict last August is not a model anyone can
    deploy.

    Why expanding rather than a rolling fixed-width window. Both are
    defensible; expanding matches how this model would actually be used
    (you have the history you have, and it grows), and it means the last
    fold is trained on nearly the whole year, which is the closest thing
    to the configuration that L6.10 will evaluate. A rolling window
    would answer a different question -- how much history is enough --
    and that question is not on this project's path.

    On the EMBARGO. `embargo_hours` discards a gap between the end of
    training and the start of validation. The standard argument for one
    is that the last training samples and the first validation samples
    are strongly correlated, so the first scored hours are partly known.
    That argument is decisive for a model with lag features, which can
    read yesterday's load directly. It is weak for this model: five
    global parameters fitted across thousands of hours cannot encode
    what happened on 31 December, so the default is zero. The knob
    exists because M7 adds a residual ML model with lag features, and at
    that point the embargo stops being optional.

    Args:
        n_samples: Length of the series.
        n_folds: Number of train/validate splits.
        spin_up_hours: Simulated-but-discarded hours before each
            validation window. Taken from the training region, so it
            costs nothing and leaks nothing -- the spin-up uses weather
            drivers and the model's own state, never validation targets.
        embargo_hours: Hours dropped between training and validation.
        min_validate_hours: Refuse to build a fold shorter than this.

    Returns:
        `n_folds` folds in time order.

    Raises:
        ValueError: If `n_folds` is below 2, if any hour count is
            negative, or if the series is too short to give every fold a
            validation window of at least `min_validate_hours` after the
            spin-up and embargo are taken out. Refused rather than
            silently returning fewer folds: a caller that asked for 5
            and got 2 will report 5.
    """
    if n_folds < 2:
        raise ValueError(
            f"n_folds must be >= 2, got {n_folds}. One fold is a single "
            "train/test split, not cross-validation, and it cannot show "
            "whether the gap is stable across the year."
        )
    for name, value in (
        ("spin_up_hours", spin_up_hours),
        ("embargo_hours", embargo_hours),
        ("min_validate_hours", min_validate_hours),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")

    # Same layout as sklearn's TimeSeriesSplit: the series is cut into
    # n_folds + 1 blocks, the first is training-only, and each
    # subsequent block is validated in turn.
    block = n_samples // (n_folds + 1)
    usable = block - embargo_hours
    if usable < min_validate_hours:
        raise ValueError(
            f"{n_samples} samples split {n_folds} ways gives {usable} validation "
            f"hours per fold after a {embargo_hours} h embargo, below the "
            f"{min_validate_hours} h minimum. Use fewer folds, or lower "
            "min_validate_hours if the shorter window is genuinely acceptable."
        )
    if block < spin_up_hours:
        raise ValueError(
            f"spin_up_hours ({spin_up_hours}) exceeds the first training block "
            f"({block} samples), so the earliest fold cannot be spun up from "
            "inside its own training data."
        )

    folds = []
    for number in range(1, n_folds + 1):
        train_stop = block * number
        validate_start = train_stop + embargo_hours
        # The last fold takes everything that is left, so no hours are
        # silently dropped off the end of the year by integer division.
        validate_stop = n_samples if number == n_folds else block * (number + 1)
        folds.append(
            TimeFold(
                number=number,
                train_start=0,
                train_stop=train_stop,
                spin_up_start=validate_start - spin_up_hours,
                validate_start=validate_start,
                validate_stop=validate_stop,
            )
        )
    logger.info(
        "%d expanding-window folds over %d samples: training %d -> %d hours, "
        "validation %d hours each (spin-up %d h, embargo %d h)",
        n_folds,
        n_samples,
        folds[0].n_train,
        folds[-1].n_train,
        folds[0].n_validate,
        spin_up_hours,
        embargo_hours,
    )
    return tuple(folds)


def random_folds(
    n_samples: int, n_folds: int = 4, seed: int = SEED
) -> tuple[npt.NDArray[np.intp], ...]:
    """Random K-fold index sets -- provided ONLY to demonstrate the leak.

    This is the split that must not be used on this data, implemented so
    that "random splits leak" can be measured rather than asserted. Pass
    its output to `neighbour_leak_test` beside a blocked split and read
    the two numbers.

    Args:
        n_samples: Length of the series.
        n_folds: Number of folds.
        seed: Seed for the permutation.

    Returns:
        One array of held-out positions per fold, sorted.

    Raises:
        ValueError: If `n_folds` is below 2 or exceeds `n_samples`.
    """
    if not 2 <= n_folds <= n_samples:
        raise ValueError(f"n_folds must be in [2, {n_samples}], got {n_folds}")
    logger.warning(
        "random_folds() is a demonstration of an INVALID split for "
        "autocorrelated data -- never score a reported result with it"
    )
    shuffled = np.random.default_rng(seed).permutation(n_samples)
    return tuple(np.sort(part) for part in np.array_split(shuffled, n_folds))


def neighbour_mean_prediction(
    values: npt.ArrayLike, held_out: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Predict held-out points from the nearest points NOT held out.

    Deliberately the stupidest model that could work: it knows no
    physics, no weather and no parameters, only that the series is
    smooth. Its purpose is to be a control. If a split lets THIS score
    well, the split is measuring smoothness rather than skill, and any
    result reported on it is unearned.

    The word NEAREST is doing the work, and getting it wrong makes the
    whole diagnostic useless. The neighbours must be drawn from the
    points the split leaves AVAILABLE, not from the series as a whole:
    an implementation that reads `values[position - 1]` regardless
    scores identically on a random split and a blocked one, because it
    is reading held-out truth in both. With availability respected, a
    randomly held-out hour still has both its immediate neighbours, while
    an hour in the middle of a held-out month has its nearest available
    neighbours a fortnight away in each direction.

    Args:
        values: The full series.
        held_out: Positions to predict. Every OTHER position is treated
            as available.

    Returns:
        One prediction per held-out position, in the order given.

    Raises:
        ValueError: If `values` has fewer than two points, if any
            position is out of range, or if every point is held out --
            with nothing available there is no prediction to make, and
            returning the series mean would quietly answer a different
            question.
    """
    series = np.asarray(values, dtype=float)
    if series.size < 2:
        raise ValueError("values must contain at least two points")
    wanted = np.asarray(held_out, dtype=np.intp)
    if wanted.size and (wanted.min() < 0 or wanted.max() >= series.size):
        raise ValueError("held_out positions must lie inside values")

    available = np.ones(series.size, dtype=bool)
    available[wanted] = False
    offsets = np.flatnonzero(available)
    if offsets.size == 0:
        raise ValueError("every point is held out; nothing is available to predict from")

    # For each held-out position, the nearest available point on each
    # side. `searchsorted` gives the insertion point, so `right` is the
    # first available position above and `right - 1` the last below.
    right = np.searchsorted(offsets, wanted)
    left = np.clip(right - 1, 0, offsets.size - 1)
    right = np.clip(right, 0, offsets.size - 1)

    left_positions, right_positions = offsets[left], offsets[right]
    has_left = left_positions < wanted
    has_right = right_positions > wanted

    left_values = series[left_positions]
    right_values = series[right_positions]
    # At either end of the series one side is missing; fall back to the
    # side that exists rather than averaging a value with itself.
    return np.where(
        has_left & has_right,
        (left_values + right_values) / 2.0,
        np.where(has_left, left_values, right_values),
    )


def neighbour_leak_test(
    values: npt.ArrayLike, held_out: Sequence[npt.ArrayLike]
) -> float:
    """Score the neighbour predictor on a candidate split.

    Returns the CV(RMSE) a model with no knowledge achieves on the
    held-out positions. Read it as a floor on how good a result the
    split can produce for free:

        low  -> the split LEAKS. Held-out points are surrounded by
                training points and are trivially interpolable.
        high -> the split holds out genuinely unseen stretches, and a
                good score on it means something.

    Args:
        values: The full series.
        held_out: One array of held-out positions per fold.

    Each fold is scored against the points that fold holds out, then the
    predictions are pooled -- the same discipline the real
    cross-validation follows. Predicting a fold's points using another
    fold's held-out values would be the very leak this function detects.

    Args:
        values: The full series.
        held_out: One array of held-out positions per fold.

    Returns:
        CV(RMSE) percent over all held-out positions pooled.
        `n_params` is 0: the neighbour predictor fits nothing.

    Raises:
        ValueError: If `held_out` is empty or holds out nothing.
    """
    if not held_out:
        raise ValueError("held_out must contain at least one fold")

    series = np.asarray(values, dtype=float)
    truths, predictions = [], []
    for part in held_out:
        positions = np.asarray(part, dtype=np.intp)
        if positions.size == 0:
            continue
        truths.append(series[positions])
        predictions.append(neighbour_mean_prediction(series, positions))
    if not truths:
        raise ValueError("held_out folds are all empty")

    return cvrmse(np.concatenate(truths), np.concatenate(predictions), n_params=0)


def cross_validate(
    fit: Callable[[TimeFold], dict[str, float]],
    predict: Callable[[dict[str, float], slice], npt.NDArray[np.float64]],
    measured: npt.ArrayLike,
    folds: Sequence[TimeFold],
    n_params: int,
    overfitting_gap_pct: float = OVERFITTING_GAP_PCT,
) -> CrossValidationResult:
    """Fit and score every fold, and report the generalisation gap.

    The caller supplies `fit` and `predict` rather than a model object,
    because the expensive part -- a full two-stage calibration -- has to
    stay under the caller's control: it decides the optimiser budget,
    the parallelism and the logging.

    ONE RULE this function enforces on the caller by its signature:
    `fit` receives a `TimeFold` and nothing else, so it cannot
    accidentally be handed the full-year optimum to start from. Warm
    starting each fold from the answer fitted on ALL of 2016 would leak
    the validation window into the fit through the starting point, and
    it is a tempting optimisation precisely because it makes every fold
    converge faster and look better.

    Args:
        fit: Calibrates on `fold.train_slice` and returns the fitted
            parameters.
        predict: Simulates one parameter set over a slice and returns
            predictions aligned with it. Called with
            `fold.simulate_slice`, so the returned array covers spin-up
            AND validation; this function drops the spin-up part.
        measured: The full measured series.
        folds: Partitions from `expanding_window_folds`.
        n_params: Calibrated parameter count, for the `n - p` correction.
        overfitting_gap_pct: Threshold carried into the result.

    Returns:
        A `CrossValidationResult`.

    Raises:
        ValueError: If `folds` is empty, or if `predict` returns an
            array whose length does not match the slice it was given --
            an off-by-one there would silently score the model against
            the wrong hours, which is the single worst failure mode in
            this file.
    """
    if not folds:
        raise ValueError("folds must not be empty")

    series = np.asarray(measured, dtype=float)
    scores = []
    names: tuple[str, ...] = ()

    for fold in folds:
        logger.info(
            "fold %d/%d: training on %d hours, validating on %d",
            fold.number,
            len(folds),
            fold.n_train,
            fold.n_validate,
        )
        parameters = fit(fold)
        names = names or tuple(parameters)

        train_predicted = predict(parameters, fold.train_slice)
        _check_length(train_predicted, fold.n_train, fold, "training")

        simulated = predict(parameters, fold.simulate_slice)
        _check_length(
            simulated, fold.n_spin_up + fold.n_validate, fold, "validation"
        )
        validate_predicted = simulated[fold.scored_offset :]

        scores.append(
            FoldScore(
                fold=fold,
                train_cvrmse_pct=cvrmse(
                    series[fold.train_slice], train_predicted, n_params
                ),
                validate_cvrmse_pct=cvrmse(
                    series[fold.validate_slice], validate_predicted, n_params
                ),
                train_nmbe_pct=nmbe(series[fold.train_slice], train_predicted, n_params),
                validate_nmbe_pct=nmbe(
                    series[fold.validate_slice], validate_predicted, n_params
                ),
                parameters=dict(parameters),
            )
        )

    result = CrossValidationResult(
        scores=tuple(scores),
        parameter_names=names,
        overfitting_gap_pct=overfitting_gap_pct,
    )
    logger.info("cross-validation finished\n%s", result.summary())
    return result


def _check_length(
    predicted: npt.NDArray[np.float64], expected: int, fold: TimeFold, window: str
) -> None:
    """Raise if a prediction array does not match its window.

    Args:
        predicted: What `predict` returned.
        expected: How many samples the slice covers.
        fold: The fold being scored, for the message.
        window: `"training"` or `"validation"`.

    Raises:
        ValueError: On any length mismatch.
    """
    if predicted.shape != (expected,):
        raise ValueError(
            f"fold {fold.number}: predict() returned {predicted.shape} for the "
            f"{window} window, expected ({expected},). Scoring would silently "
            "compare the model against the wrong hours."
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    # A year of hourly data with the two properties that matter: a
    # strong daily cycle and strong hour-to-hour persistence. Nothing
    # here is a building -- the leak being demonstrated is a property of
    # the SPLIT, not of any model, and it survives any smooth series.
    HOURS = 8760
    rng = np.random.default_rng(SEED)
    hour_of_day = np.arange(HOURS) % 24
    seasonal = 300.0 * np.sin(2 * np.pi * np.arange(HOURS) / HOURS - np.pi / 2)
    daily = 200.0 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    noise = np.cumsum(rng.normal(0.0, 8.0, HOURS))
    noise -= np.linspace(0.0, noise[-1], HOURS)  # keep it stationary
    load_kw = 1500.0 + seasonal + daily + noise

    logger.info("--- 1. what a random split holds out ---")
    scattered = random_folds(HOURS, n_folds=4)
    blocked = expanding_window_folds(HOURS, n_folds=4)
    blocked_positions = [
        np.arange(fold.validate_start, fold.validate_stop) for fold in blocked
    ]

    logger.info("--- 2. the control: a model that knows nothing ---")
    leak = {
        label: neighbour_leak_test(load_kw, held_out)
        for label, held_out in (
            ("random 4-fold", scattered),
            ("blocked 4-fold", blocked_positions),
        )
    }
    for label, score in leak.items():
        logger.info("%-16s neighbour-mean CV(RMSE) = %6.2f%%", label, score)
    logger.info(
        "  the same knowledge-free model is %.0fx worse under the blocked "
        "split. Under the random one it scores %.2f%% -- inside G14's 30%% "
        "hourly limit -- while knowing nothing whatsoever about the building.",
        leak["blocked 4-fold"] / leak["random 4-fold"],
        leak["random 4-fold"],
    )

    logger.info("--- 3. the folds a blocked split actually produces ---")
    for fold in blocked:
        logger.info(
            "  fold %d: train [%5d, %5d)  spin-up %d h  validate [%5d, %5d)",
            fold.number,
            fold.train_start,
            fold.train_stop,
            fold.n_spin_up,
            fold.validate_start,
            fold.validate_stop,
        )

    logger.info("--- 4. scoring a deliberately overfitted model ---")
    # `fit` returns a constant equal to the MEAN OF ITS OWN TRAINING
    # WINDOW -- the simplest model that can overfit a trend. On a series
    # with a seasonal swing, each fold's training mean is stale by the
    # time the validation window arrives, and the gap shows it.
    def fit_training_mean(fold: TimeFold) -> dict[str, float]:
        """Fit the one parameter this toy model has."""
        return {"level_kw": float(np.mean(load_kw[fold.train_slice]))}

    def predict_constant(
        parameters: dict[str, float], window: slice
    ) -> npt.NDArray[np.float64]:
        """Predict that constant over any window."""
        return np.full(len(load_kw[window]), parameters["level_kw"])

    result = cross_validate(
        fit_training_mean, predict_constant, load_kw, blocked, n_params=1
    )
    logger.info("\n%s", result.summary())
    logger.info(
        "  level_kw across folds: %.1f to %.1f kW -- a parameter that moves "
        "this much between folds is tracking the season, not the building",
        *result.parameter_stability()["level_kw"],
    )

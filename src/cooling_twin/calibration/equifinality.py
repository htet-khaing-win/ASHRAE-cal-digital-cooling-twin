"""Equifinality -- many different parameter sets, one indistinguishable fit.

L6.7 returns ONE parameter vector and a CV(RMSE). That number is real.
The vector is a claim, and it is a much weaker claim than it looks: on a
model with five parameters and one output series, there are usually
whole FAMILIES of parameter sets that reproduce the measured load
equally well. Beven and Binley named the survivors of that family
"behavioural" parameter sets (GLUE, 1992) -- behavioural meaning "cannot
be rejected on the evidence available", which is a very different
statement from "correct".

Why this module exists rather than a paragraph of caution:

    A calibrated twin is used to answer questions the calibration data
    never asked. "What if we upgrade the envelope?" "What if we cut the
    ventilation rate?" Two parameter sets with the SAME CV(RMSE) can
    give opposite answers to those questions, because they attribute
    the same measured load to different physical causes. One says the
    building leaks; the other says the fans are oversized. Fit quality
    cannot separate them. A retrofit budget can.

So the deliverable here is not a better fit. It is an honest interval
around the fitted parameters, a list of which parameters trade off
against which, and -- the part with money attached -- the SPREAD in the
downstream answer across parameter sets the data cannot tell apart.

Three mitigations exist, and this module supports reasoning about all
three rather than implementing a fix:

    1. Tighter bounds       -- narrow the box on physical grounds
                               (`calibration/bounds.py`, ADR-013).
    2. Fewer parameters     -- screen before calibrating
                               (`calibration/sensitivity.py`, L6.5).
    3. More targets         -- fit against a second measured series, so
                               a trade-off that is invisible in one
                               output becomes visible in the other.

Mitigation 3 is the only one that adds information; the first two only
stop the search from spending information it does not have.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import qmc

from cooling_twin import SEED
from cooling_twin.calibration.optimize import (
    LOCAL_STEP_FRACTION,
    PINNED_TOLERANCE,
    _FiniteObjective,
)

logger = logging.getLogger(__name__)

# A parameter set is behavioural when its objective is within this
# RELATIVE margin of the best one found. 5% is not a statistical
# threshold -- there is no sampling distribution here -- it is a
# statement about resolution: L6.7b measured the entire L-BFGS-B
# refinement stage buying 0.04% of objective, and the objective itself
# is an adaptive-step ODE solve whose surface is slightly discontinuous
# at the solver's tolerance. Differences of a few percent are therefore
# not differences this project can defend. Report the CV(RMSE) span the
# threshold corresponds to on real data, every time -- that is the
# number a reviewer will actually weigh.
DEFAULT_BEHAVIOURAL_TOLERANCE = 0.05

# Restarts to run. Cost is `n_starts` local refinements, each a few
# hundred objective evaluations, so on this project this is a
# wall-clock decision: at ~1 s per evaluation (L6.7b) twelve starts is
# roughly an hour serial and a few minutes across cores. Twelve is
# enough to show a ridge and not enough to map one.
DEFAULT_N_STARTS = 12

# Iteration cap for each local refinement, matching `calibrate()`'s.
LOCAL_MAXITER = 200

# Spread verdict thresholds, as a fraction of each parameter's bound
# width. Below 10% the behavioural sets all agree on the value, so the
# data identifies it. Above 50% the parameter takes nearly any value in
# its box without harming the fit, so the calibration has not measured
# it at all and reporting it to three decimals is a lie of precision.
IDENTIFIED_SPAN_FRACTION = 0.10
UNIDENTIFIED_SPAN_FRACTION = 0.50

# Below this many behavioural sets, a spread is not evidence: the span
# of one point is zero by construction, which reads as "perfectly
# identified" and means nothing.
MIN_SETS_FOR_SPREAD = 3

# Correlation needs more points than a spread does, and how many more
# depends on the PARAMETER COUNT -- see `minimum_sets_for_correlation`.
# Two spare points above the dimension is the minimum at which a
# correlation is measuring the model rather than the sample size.
CORRELATION_SETS_ABOVE_DIMENSION = 2

# |correlation| above which two parameters are reported as compensating.
# 0.8 is a reporting threshold, not a physical one.
DEFAULT_COMPENSATION_THRESHOLD = 0.8


@dataclass(frozen=True)
class BehaviouralSet:
    """One parameter set that the calibration data cannot reject.

    Attributes:
        parameters: The refined vector, in the study's parameter order.
        objective: Objective value there.
        start: Where the refinement started -- kept because a family of
            optima found from a single start is a different (and much
            weaker) finding than the same family found from many.
        origin: `"reference"` for the calibrated optimum under test, or
            `"restart-<i>"`.
        message: The local optimiser's termination message.
    """

    parameters: tuple[float, ...]
    objective: float
    start: tuple[float, ...]
    origin: str
    message: str


@dataclass(frozen=True)
class ParameterSpread:
    """How much one parameter moves across the behavioural sets.

    Attributes:
        name: Parameter name.
        minimum: Smallest behavioural value.
        maximum: Largest behavioural value.
        span_fraction: `(maximum - minimum) / (upper - lower)` -- the
            share of its own physical range the parameter roams over
            without the fit noticing.
        pinned_at: `"lower"`, `"upper"` or `None` -- set when EVERY
            behavioural set leaves the parameter on the same bound.
        verdict: `"identified"`, `"weakly identified"`, `"unidentified"`
            or `"bound-limited"`, cut at `IDENTIFIED_SPAN_FRACTION` and
            `UNIDENTIFIED_SPAN_FRACTION`.
    """

    name: str
    minimum: float
    maximum: float
    span_fraction: float
    verdict: str
    pinned_at: str | None = None

    @property
    def span(self) -> float:
        """Absolute width of the behavioural range, in the parameter's units."""
        return self.maximum - self.minimum


@dataclass(frozen=True)
class OutcomeSpread:
    """The disagreement between behavioural sets about a downstream answer.

    This is the reason equifinality is a finding rather than a
    curiosity. `values` are one answer per behavioural set -- a
    predicted retrofit saving, a peak load, an annual energy -- all
    produced by parameter sets the calibration cannot rank.

    Attributes:
        values: One outcome per behavioural set, in study order.
        minimum: Smallest outcome.
        maximum: Largest outcome.
        median: Median outcome.
        reference: Outcome at the calibrated (reference) set, when that
            set is in the study.
    """

    values: tuple[float, ...]
    minimum: float
    maximum: float
    median: float
    reference: float | None

    @property
    def spread(self) -> float:
        """`maximum - minimum` -- the width of the answer the twin can give."""
        return self.maximum - self.minimum

    def summary(self, unit: str = "") -> str:
        """One-line human-readable spread, for logs and reports."""
        suffix = f" {unit}" if unit else ""
        reference = "n/a" if self.reference is None else f"{self.reference:.3f}{suffix}"
        return (
            f"{len(self.values)} behavioural sets: "
            f"{self.minimum:.3f}{suffix} to {self.maximum:.3f}{suffix} "
            f"(median {self.median:.3f}{suffix}, calibrated {reference}, "
            f"spread {self.spread:.3f}{suffix})"
        )


@dataclass(frozen=True)
class EquifinalityStudy:
    """The outcome of probing one calibration for indistinguishable rivals.

    Attributes:
        parameter_names: Parameter names, in vector order.
        bounds: `(lower, upper)` per parameter, same order.
        sets: Behavioural sets, best objective first.
        rejected: Refinements that finished ABOVE the threshold. Kept
            because "10 of 12 restarts reached the same fit" and "2 of
            12 did" are opposite statements about the surface.
        reference_objective: Objective at the calibrated vector under
            test.
        best_objective: Best objective found anywhere in the study.
        tolerance: Relative margin defining behavioural.
        threshold: `best_objective * (1 + tolerance)`.
        seed: Seed for the restart sampler.
        start_spread: Fraction of each bound width the restarts were
            drawn from -- part of the result, not a setting, because a
            spread measured from a concentrated sample answers a
            different question from one measured over the whole box.
        n_evaluations: Total objective calls across all refinements.
        elapsed_seconds: Wall clock.
    """

    parameter_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    sets: tuple[BehaviouralSet, ...]
    rejected: tuple[BehaviouralSet, ...]
    reference_objective: float
    best_objective: float
    tolerance: float
    threshold: float
    seed: int
    n_evaluations: int
    elapsed_seconds: float
    start_spread: float = 1.0

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Behavioural parameter vectors as an `(n_sets, n_params)` array."""
        return np.array([entry.parameters for entry in self.sets], dtype=float)

    @property
    def reference_set(self) -> BehaviouralSet | None:
        """The refined calibrated vector, if it survived the threshold."""
        for entry in self.sets:
            if entry.origin == "reference":
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record, for the run log."""
        return {
            "tolerance": self.tolerance,
            "threshold": self.threshold,
            "reference_objective": self.reference_objective,
            "best_objective": self.best_objective,
            "n_behavioural": len(self.sets),
            "n_rejected": len(self.rejected),
            "n_evaluations": self.n_evaluations,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "seed": self.seed,
            "start_spread": self.start_spread,
            "spread": {
                name: {
                    "min": spread.minimum,
                    "max": spread.maximum,
                    "span_fraction": spread.span_fraction,
                    "verdict": spread.verdict,
                    "pinned_at": spread.pinned_at,
                }
                for name, spread in parameter_spread(self).items()
            },
            "parameter_names": list(self.parameter_names),
            "bounds": {
                name: list(bound)
                for name, bound in zip(self.parameter_names, self.bounds, strict=True)
            },
            # EVERY refinement, not just the surviving ones, and the
            # rejected ones in full rather than as a count. Two reasons.
            # They describe the SURFACE -- a cluster just above the
            # threshold means the behavioural region was under-sampled
            # and `n_starts` should rise, while a scatter of far worse
            # values means the restarts are landing in other basins,
            # which is the empirical case for L6.7's global stage. And
            # they make the artifact re-analysable: `from_dict` plus
            # `rethreshold` can re-partition the study at another
            # tolerance months later without re-running a single
            # simulation. An artifact that records only the answer
            # cannot be re-questioned.
            "candidates": [
                {
                    "origin": entry.origin,
                    "objective": entry.objective,
                    "behavioural": behavioural,
                    "message": entry.message,
                    "parameters": dict(
                        zip(self.parameter_names, entry.parameters, strict=True)
                    ),
                    "start": dict(zip(self.parameter_names, entry.start, strict=True)),
                }
                for entries, behavioural in ((self.sets, True), (self.rejected, False))
                for entry in entries
            ],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> EquifinalityStudy:
        """Rebuild a study from `to_dict`'s output.

        Lossless for everything the analysis functions use, which is
        what makes a written study re-analysable rather than merely
        readable.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The reconstructed study.

        Raises:
            KeyError: If the record is missing a required field --
                raised rather than defaulted, because a study silently
                rebuilt with an empty `rejected` list would report a
                different threshold sensitivity than the run it claims
                to be.
        """
        names = tuple(record["parameter_names"])
        entries = {True: [], False: []}  # type: dict[bool, list[BehaviouralSet]]
        for candidate in record["candidates"]:
            entries[bool(candidate["behavioural"])].append(
                BehaviouralSet(
                    parameters=tuple(
                        float(candidate["parameters"][name]) for name in names
                    ),
                    objective=float(candidate["objective"]),
                    start=tuple(float(candidate["start"][name]) for name in names),
                    origin=str(candidate["origin"]),
                    message=str(candidate["message"]),
                )
            )
        return cls(
            parameter_names=names,
            bounds=tuple((float(low), float(high)) for low, high in
                         (record["bounds"][name] for name in names)),
            sets=tuple(sorted(entries[True], key=lambda entry: entry.objective)),
            rejected=tuple(entries[False]),
            reference_objective=float(record["reference_objective"]),
            best_objective=float(record["best_objective"]),
            tolerance=float(record["tolerance"]),
            threshold=float(record["threshold"]),
            seed=int(record["seed"]),
            n_evaluations=int(record["n_evaluations"]),
            elapsed_seconds=float(record["elapsed_seconds"]),
            start_spread=float(record["start_spread"]),
        )

    def rethreshold(self, tolerance: float) -> EquifinalityStudy:
        """Re-partition the SAME refinements at a different tolerance.

        Costs nothing, and that is the reason `rejected` is retained
        rather than discarded at collection time: the behavioural
        threshold is a judgement, so a result that depends on it must be
        shown at more than one value. Recollecting instead would spend
        another few hundred simulations to answer a question the
        existing ones already answer.

        Args:
            tolerance: The new relative margin.

        Returns:
            A new study over the same candidates.

        Raises:
            ValueError: If `tolerance` is not positive.
        """
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        candidates = [*self.sets, *self.rejected]
        threshold = self.best_objective * (1.0 + tolerance)
        return replace(
            self,
            sets=tuple(
                sorted(
                    (entry for entry in candidates if entry.objective <= threshold),
                    key=lambda entry: entry.objective,
                )
            ),
            rejected=tuple(
                entry for entry in candidates if entry.objective > threshold
            ),
            tolerance=tolerance,
            threshold=threshold,
        )

    def summary(self) -> str:
        """Multi-line human-readable summary, for logs and notebooks."""
        lines = [
            f"{len(self.sets)} behavioural of {len(self.sets) + len(self.rejected)} "
            f"refinements, objective <= {self.threshold:.6f} "
            f"(best {self.best_objective:.6f}, calibrated "
            f"{self.reference_objective:.6f}, tolerance {self.tolerance:.1%})",
            f"{self.n_evaluations} evaluations in {self.elapsed_seconds:.1f}s",
        ]
        for name, spread in parameter_spread(self).items():
            lines.append(
                f"  {name:<28} {spread.minimum:10.3f} .. {spread.maximum:10.3f}  "
                f"{spread.span_fraction:6.1%} of range  {spread.verdict}"
                + (f" [{spread.pinned_at} bound]" if spread.pinned_at else "")
            )
        return "\n".join(lines)


class _LocalRefiner:
    """Picklable "refine from this start" callable, for the process pool.

    Same reason as `optimize._FiniteObjective`: restarts are run across
    processes, a process pool pickles what it is given, and a closure
    over `bounds` and `steps` is not picklable. An instance holding
    arrays and the (already picklable) objective is.

    Attributes:
        objective: The wrapped objective.
    """

    def __init__(
        self,
        objective: Callable[[npt.NDArray[np.float64]], float],
        bounds: tuple[tuple[float, float], ...],
        steps: npt.NDArray[np.float64],
    ) -> None:
        self.objective = _FiniteObjective(objective)
        self._bounds = bounds
        self._steps = steps

    def __call__(
        self, start: npt.NDArray[np.float64]
    ) -> tuple[tuple[float, ...], float, str, int]:
        """Run one bounded L-BFGS-B refinement.

        Args:
            start: Starting vector, inside the bounds.

        Returns:
            `(parameters, objective, message, n_evaluations)`.
        """
        result = minimize(
            self.objective,
            x0=np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=self._bounds,
            options={"eps": self._steps, "maxiter": LOCAL_MAXITER},
        )
        return (
            tuple(float(value) for value in result.x),
            float(result.fun),
            str(result.message),
            int(result.nfev),
        )


def _validated_bounds(
    bounds: Mapping[str, tuple[float, float]],
) -> tuple[tuple[str, ...], tuple[tuple[float, float], ...]]:
    """Check a bounds mapping and split it into names and pairs.

    Args:
        bounds: Ordered mapping of parameter name to `(lower, upper)`.

    Returns:
        `(names, pairs)` in insertion order.

    Raises:
        ValueError: If `bounds` is empty, or any bound is non-finite or
            not strictly increasing.
    """
    if not bounds:
        raise ValueError("bounds must contain at least one parameter")
    names = tuple(bounds)
    pairs = tuple((float(low), float(high)) for low, high in bounds.values())
    for name, (low, high) in zip(names, pairs, strict=True):
        if not np.isfinite([low, high]).all():
            raise ValueError(f"{name}: bounds must be finite, got ({low}, {high})")
        if not low < high:
            raise ValueError(
                f"{name}: lower bound ({low}) must be strictly less than upper ({high})"
            )
    return names, pairs


def collect_behavioural_sets(
    objective: Callable[[npt.NDArray[np.float64]], float],
    bounds: Mapping[str, tuple[float, float]],
    reference_parameters: npt.ArrayLike,
    tolerance: float = DEFAULT_BEHAVIOURAL_TOLERANCE,
    n_starts: int = DEFAULT_N_STARTS,
    seed: int = SEED,
    workers: int = 1,
    start_spread: float = 1.0,
) -> EquifinalityStudy:
    """Find parameter sets that fit as well as the calibrated one.

    Each restart is a bounded local refinement from a Latin-hypercube
    start, plus one refinement from the calibrated vector itself. Every
    refinement that lands within `tolerance` of the best objective found
    is behavioural: the data cannot rank it below the reported answer.

    Why local refinements from scattered starts rather than a second
    global search. Differential evolution is designed to CONVERGE -- it
    collapses its population onto one basin and reports the winner,
    which is exactly the information being questioned here. The
    question is not "what is the best fit" (L6.7 answered that) but
    "what else fits this well", so the search has to be one that keeps
    its answers separate.

    Why the reference vector is refined too rather than inserted
    directly: if the calibrated answer does not survive its own
    threshold, the study is measuring a different problem, and that has
    to be visible instead of assumed away.

    Args:
        objective: Maps a parameter vector, ordered as `bounds`, to the
            scalar minimised. Must be deterministic. With
            `workers != 1` it must also be picklable -- see
            `optimize.calibrate` for what that rules out.
        bounds: Ordered mapping of parameter name to `(lower, upper)`.
            Must be the SAME box the calibration used; a wider box here
            measures the spread of a different calibration.
        reference_parameters: The calibrated vector under test, in the
            same order. Any array-like of the right length.
        tolerance: Relative margin defining behavioural.
        n_starts: Number of scattered restarts, additional to the
            reference refinement.
        seed: Seed for the Latin-hypercube sampler.
        workers: `1` runs serially; any other value runs the
            refinements across a process pool, `-1` meaning every core.
            Restarts are independent and collected by index, so the
            result does not depend on completion order.
        start_spread: Fraction of each bound width the restarts are
            drawn from, centred on the reference and clipped to the
            bounds. `1.0` -- the default -- scatters over the whole box
            and asks "is there a DISTANT rival that fits as well". A
            small value concentrates the budget near the reported
            optimum and asks the different question "how wide is the
            ridge the optimum sits on". Both are legitimate and they
            are not interchangeable: on a surface whose behavioural
            region is small, a whole-box sample spends nearly all its
            refinements in other basins and reports a handful of
            behavioural sets -- too few to correlate -- while a
            concentrated sample maps the trade-offs but says nothing
            about distant rivals. Run both and report both.

    Returns:
        An `EquifinalityStudy`.

    Raises:
        ValueError: If `bounds` is invalid, `tolerance` is not positive,
            `n_starts` is negative, `reference_parameters` has the wrong
            length or lies outside the bounds, or the objective is not
            finite at the reference vector.
    """
    names, pairs = _validated_bounds(bounds)
    if tolerance <= 0.0:
        raise ValueError(
            f"tolerance must be > 0, got {tolerance}. A zero tolerance admits "
            "only exact ties, which floating-point arithmetic never produces -- "
            "the study would report a single behavioural set and look reassuring."
        )
    if n_starts < 0:
        raise ValueError(f"n_starts must be >= 0, got {n_starts}")
    if not 0.0 < start_spread <= 1.0:
        raise ValueError(
            f"start_spread must be in (0, 1], got {start_spread}. Zero would "
            "start every refinement at the reference, which finds one point "
            "and calls it a family."
        )

    reference = np.asarray(reference_parameters, dtype=float)
    if reference.shape != (len(names),):
        raise ValueError(
            f"reference_parameters has length {reference.size}, but bounds "
            f"describes {len(names)} parameters {names}"
        )
    lower = np.array([low for low, _ in pairs])
    upper = np.array([high for _, high in pairs])
    outside = [
        name
        for name, value, low, high in zip(names, reference, lower, upper, strict=True)
        if value < low or value > high
    ]
    if outside:
        raise ValueError(
            f"reference_parameters lie outside the bounds for {outside}. The "
            "study must use the box the calibration used, or the spread it "
            "reports belongs to a different calibration."
        )

    reference_objective = float(objective(reference))
    if not np.isfinite(reference_objective):
        raise ValueError(
            f"objective is not finite at the reference vector: {reference_objective}"
        )

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    # The window shrinks toward the reference by `start_spread` of the
    # room available on EACH side, rather than as a symmetric band of
    # fixed width clipped to the box. Two reasons. At 1.0 this returns
    # the box exactly, so the default keeps meaning "the whole box" --
    # a symmetric band clipped to the bounds does not, and on this
    # building (four parameters sitting near a bound) it silently
    # sampled as little as half the box while claiming to sample all of
    # it. And a reference near a bound has little room on one side and
    # plenty on the other; shrinking proportionally respects that
    # instead of spending half the window outside the box.
    window_lower = reference - start_spread * (reference - lower)
    window_upper = reference + start_spread * (upper - reference)
    scattered = (
        qmc.scale(sampler.random(n=n_starts), window_lower, window_upper)
        if n_starts
        else np.empty((0, len(names)))
    )
    starts = np.vstack([reference.reshape(1, -1), scattered])
    origins = ["reference", *(f"restart-{index}" for index in range(n_starts))]

    steps = np.array([LOCAL_STEP_FRACTION * (high - low) for low, high in pairs])
    refiner = _LocalRefiner(objective, pairs, steps)

    logger.info(
        "equifinality probe: %d refinements (1 reference + %d restarts over %.0f%% "
        "of each bound width), %d parameters",
        len(starts),
        n_starts,
        100 * start_spread,
        len(names),
    )
    started = time.perf_counter()
    if workers == 1:
        outcomes = [refiner(start) for start in starts]
    else:
        max_workers = os.cpu_count() if workers < 0 else workers
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(refiner, list(starts)))
    elapsed = time.perf_counter() - started

    candidates = [
        BehaviouralSet(
            parameters=parameters,
            objective=value,
            start=tuple(float(component) for component in start),
            origin=origin,
            message=message,
        )
        for start, origin, (parameters, value, message, _) in zip(
            starts, origins, outcomes, strict=True
        )
    ]
    best_objective = min(entry.objective for entry in candidates)
    if best_objective < reference_objective * (1.0 - tolerance):
        logger.warning(
            "a restart BEAT the calibrated optimum (%.6f < %.6f). The reported "
            "calibration is not the best fit in its own box -- fix that before "
            "reading anything into the spread below.",
            best_objective,
            reference_objective,
        )
    threshold = best_objective * (1.0 + tolerance)

    behavioural = tuple(
        sorted(
            (entry for entry in candidates if entry.objective <= threshold),
            key=lambda entry: entry.objective,
        )
    )
    rejected = tuple(entry for entry in candidates if entry.objective > threshold)

    study = EquifinalityStudy(
        parameter_names=names,
        bounds=pairs,
        sets=behavioural,
        rejected=rejected,
        reference_objective=reference_objective,
        best_objective=best_objective,
        tolerance=tolerance,
        threshold=threshold,
        seed=seed,
        start_spread=start_spread,
        n_evaluations=sum(int(outcome[3]) for outcome in outcomes),
        elapsed_seconds=elapsed,
    )
    logger.info("equifinality probe finished\n%s", study.summary())
    return study


def parameter_spread(study: EquifinalityStudy) -> dict[str, ParameterSpread]:
    """Measure how far each parameter roams across the behavioural sets.

    The span is expressed as a FRACTION of the parameter's own bound
    width, because the raw span is not comparable between a parameter
    spanning 0.3-3.0 and one spanning 0.5-180: 5 units is the whole
    story for the first and rounding error for the second.

    A narrow span has TWO causes and they are opposite findings. Either
    the data holds the parameter in place -- identified -- or the BOX
    does, every behavioural set piled against the same bound because the
    fit wants to keep going and cannot. The second reads as a span of
    zero, which is the most confident-looking output this function can
    produce and means the least. It is reported as `"bound-limited"`
    with `pinned_at` set, never as `"identified"`.

    Args:
        study: A completed study.

    Returns:
        `{name: ParameterSpread}` in the study's parameter order.
    """
    if len(study.sets) < MIN_SETS_FOR_SPREAD:
        logger.warning(
            "only %d behavioural set(s): a span computed from this many points "
            "is not evidence of identifiability. Raise n_starts, or widen the "
            "tolerance if the surface is genuinely that sharp.",
            len(study.sets),
        )

    matrix = study.matrix
    spreads: dict[str, ParameterSpread] = {}
    for index, (name, (low, high)) in enumerate(
        zip(study.parameter_names, study.bounds, strict=True)
    ):
        column = matrix[:, index] if matrix.size else np.array([np.nan])
        minimum = float(np.min(column))
        maximum = float(np.max(column))
        span_fraction = (maximum - minimum) / (high - low)

        margin = PINNED_TOLERANCE * (high - low)
        pinned_at: str | None = None
        if maximum <= low + margin:
            pinned_at = "lower"
        elif minimum >= high - margin:
            pinned_at = "upper"

        if pinned_at is not None and span_fraction <= IDENTIFIED_SPAN_FRACTION:
            verdict = "bound-limited"
        elif span_fraction <= IDENTIFIED_SPAN_FRACTION:
            verdict = "identified"
        elif span_fraction >= UNIDENTIFIED_SPAN_FRACTION:
            verdict = "unidentified"
        else:
            verdict = "weakly identified"
        spreads[name] = ParameterSpread(
            name=name,
            minimum=minimum,
            maximum=maximum,
            span_fraction=span_fraction,
            verdict=verdict,
            pinned_at=pinned_at,
        )
    return spreads


def minimum_sets_for_correlation(n_params: int) -> int:
    """Behavioural sets needed before a correlation means anything.

    `n` points always lie inside an `(n-1)`-dimensional affine subspace
    of the parameter space. When `n` is at or below the number of
    parameters, that subspace is a proper slice of the box and EVERY
    pair of coordinates comes out strongly correlated -- not because the
    parameters trade off, but because three points in five dimensions
    have nowhere else to be. Reporting "r = +1.000" from such a sample
    would dress up the sample size as a physical finding, and it is the
    kind of mistake that survives review because the number looks
    decisive.

    The rule is therefore `n_params + CORRELATION_SETS_ABOVE_DIMENSION`:
    enough points that the family has room to be uncorrelated if it
    wants to be.

    Args:
        n_params: Number of calibrated parameters.

    Returns:
        The minimum behavioural-set count.

    Raises:
        ValueError: If `n_params` is not positive.
    """
    if n_params < 1:
        raise ValueError(f"n_params must be >= 1, got {n_params}")
    return max(MIN_SETS_FOR_SPREAD, n_params + CORRELATION_SETS_ABOVE_DIMENSION)


def parameter_correlations(study: EquifinalityStudy) -> npt.NDArray[np.float64]:
    """Correlation between parameters ACROSS the behavioural sets.

    This is the mechanism behind a wide spread, and it separates two
    findings that a spread alone conflates. A parameter can roam because
    the model does not care about it at all (inert -- it correlates with
    nothing), or because another parameter moves with it to keep the
    predicted load the same (compensating -- |r| near 1). The first is a
    screening failure; the second is equifinality proper, and only the
    second makes a downstream answer ambiguous.

    Args:
        study: A completed study.

    Returns:
        An `(n_params, n_params)` correlation matrix. Entries involving
        a parameter that is constant across the behavioural sets are
        `nan` -- correlation with a constant is undefined, and returning
        0.0 there would read as "independent", which is a different
        claim.

    Raises:
        ValueError: If there are fewer behavioural sets than the
            parameter count requires -- see
            `minimum_sets_for_correlation`.
    """
    required = minimum_sets_for_correlation(len(study.parameter_names))
    if len(study.sets) < required:
        raise ValueError(
            f"need at least {required} behavioural sets to correlate "
            f"{len(study.parameter_names)} parameters, got {len(study.sets)}. "
            "n points lie in an (n-1)-dimensional plane, so with n at or below "
            "the parameter count every pair comes out near +-1 whatever the "
            "model does -- the trade-off would be an artifact of the sample "
            "size. Raise n_starts, or widen the tolerance if the surface is "
            "genuinely that sharp."
        )
    matrix = study.matrix
    constant = np.std(matrix, axis=0) == 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = np.corrcoef(matrix, rowvar=False)
    correlations = np.asarray(correlations, dtype=float)
    correlations[constant, :] = np.nan
    correlations[:, constant] = np.nan
    return correlations


def compensating_pairs(
    study: EquifinalityStudy,
    threshold: float = DEFAULT_COMPENSATION_THRESHOLD,
) -> tuple[tuple[str, str, float], ...]:
    """List parameter pairs that trade off against each other.

    Args:
        study: A completed study.
        threshold: Report pairs with `|r| >= threshold`.

    Returns:
        `(name_a, name_b, r)` tuples, strongest first.

    Raises:
        ValueError: If `threshold` is not in (0, 1], or from
            `parameter_correlations`.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    correlations = parameter_correlations(study)
    names = study.parameter_names
    pairs = [
        (names[row], names[column], float(correlations[row, column]))
        for row in range(len(names))
        for column in range(row + 1, len(names))
        if np.isfinite(correlations[row, column])
        and abs(correlations[row, column]) >= threshold
    ]
    return tuple(sorted(pairs, key=lambda pair: -abs(pair[2])))


def most_divergent_pair(
    study: EquifinalityStudy, parameter: str
) -> tuple[BehaviouralSet, BehaviouralSet]:
    """The two behavioural sets that disagree most about one parameter.

    This is the exhibit, not a statistic: two complete parameter sets,
    both admissible on the evidence, telling different stories about the
    same building.

    Args:
        study: A completed study.
        parameter: Which parameter to spread on.

    Returns:
        `(lowest, highest)` by that parameter's value.

    Raises:
        KeyError: If `parameter` is not in the study.
        ValueError: If there are fewer than two behavioural sets.
    """
    if parameter not in study.parameter_names:
        raise KeyError(
            f"{parameter!r} is not a calibrated parameter: {study.parameter_names}"
        )
    if len(study.sets) < 2:
        raise ValueError(
            f"need at least 2 behavioural sets to show a divergence, got "
            f"{len(study.sets)}"
        )
    index = study.parameter_names.index(parameter)
    ordered = sorted(study.sets, key=lambda entry: entry.parameters[index])
    return ordered[0], ordered[-1]


def outcome_spread(
    study: EquifinalityStudy,
    outcome: Callable[[npt.NDArray[np.float64]], float],
) -> OutcomeSpread:
    """Evaluate a downstream answer at every behavioural set.

    The point of the whole module. `outcome` is whatever the twin is
    actually going to be used for -- predicted saving from a retrofit,
    annual energy, peak load -- evaluated at each parameter set that the
    calibration cannot rank. The width of the result is the honest
    uncertainty in that answer from equifinality ALONE: it excludes
    measurement error, weather-year variation and model structure, so it
    is a lower bound on the total, never an error bar.

    Args:
        study: A completed study.
        outcome: Maps a parameter vector to one scalar answer.

    Returns:
        An `OutcomeSpread`.

    Raises:
        ValueError: If the study has no behavioural sets, or `outcome`
            returns a non-finite value -- silently dropping those would
            narrow the reported spread, which is the one direction of
            error this function must never make.
    """
    if not study.sets:
        raise ValueError("study has no behavioural sets to evaluate")

    values = []
    for entry in study.sets:
        value = float(outcome(np.asarray(entry.parameters, dtype=float)))
        if not np.isfinite(value):
            raise ValueError(
                f"outcome is not finite at behavioural set {entry.origin!r}: "
                f"{value}. Dropping it would narrow the reported spread."
            )
        values.append(value)

    reference_entry = study.reference_set
    reference_value = (
        float(outcome(np.asarray(reference_entry.parameters, dtype=float)))
        if reference_entry is not None
        else None
    )
    return OutcomeSpread(
        values=tuple(values),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        median=float(np.median(values)),
        reference=reference_value,
    )


# --- demo -----------------------------------------------------------------
#
# A KNOWN-ANSWER surface with the three patterns this module has to tell
# apart, built so that every expected number is derivable by hand.
#
#     objective(g, m, t, r) = FLOOR + (g/400 + m/180 - 1)^2 + (t - 22)^2
#
#   * g and m enter ONLY through their sum, so any pair on the line
#     g/400 + m/180 = 1 fits exactly as well: a ridge, correlation -1.
#     This is the real model's structure in miniature -- internal gain
#     and ventilation flow both scale total load, and one measured load
#     series cannot say which produced it.
#   * t is identified: a 1 K error costs the whole tolerance budget.
#   * r appears nowhere. Inert, not compensating.
#
# FLOOR stands in for the irreducible error every real objective has.
# Without it the best objective is 0 and a RELATIVE tolerance is
# meaningless -- which is itself worth knowing before running this
# against anything.

DEMO_BOUNDS = {
    "internal_gain_w_per_m2": (1.0, 400.0),
    "vent_flow_kg_per_s": (0.5, 180.0),
    "t_setpoint_c": (20.0, 26.0),
    "r_internal_ratio": (2.0, 20.0),
}
DEMO_FLOOR = 0.20
DEMO_GAIN_SCALE = 400.0
DEMO_VENT_SCALE = 180.0
DEMO_SETPOINT_C = 22.0
DEMO_VENT_CUT = 0.30


def _demo_objective(vector: npt.NDArray[np.float64]) -> float:
    """Known-answer ridge surface. Minimum `DEMO_FLOOR` on a whole line."""
    gain, vent, setpoint, _inert = (float(value) for value in vector)
    scale_error = gain / DEMO_GAIN_SCALE + vent / DEMO_VENT_SCALE - 1.0
    return DEMO_FLOOR + scale_error**2 + (setpoint - DEMO_SETPOINT_C) ** 2


def _demo_retrofit_saving_pct(vector: npt.NDArray[np.float64]) -> float:
    """Predicted saving from cutting ventilation flow by `DEMO_VENT_CUT`.

    Total load is `gain_share + vent_share` and the retrofit removes
    30% of the ventilation share, so the saving is
    `30% * vent_share / (gain_share + vent_share)`. On the ridge the
    denominator is 1, so the answer is simply `30% * vent_share` -- and
    `vent_share` runs from ~0 to ~1 along a ridge of equally good fits.
    """
    gain, vent, _setpoint, _inert = (float(value) for value in vector)
    gain_share = gain / DEMO_GAIN_SCALE
    vent_share = vent / DEMO_VENT_SCALE
    return 100.0 * DEMO_VENT_CUT * vent_share / (gain_share + vent_share)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    logger.info("--- 1. one calibrated answer ---")
    # A point exactly on the ridge: half the load from gains, half from
    # ventilation. This stands in for L6.7's reported optimum.
    calibrated = np.array([200.0, 90.0, 22.0, 11.0])
    logger.info(
        "calibrated: gain %.1f W/m2, vent %.1f kg/s, setpoint %.1f C, ratio %.1f "
        "-> objective %.6f",
        *calibrated,
        _demo_objective(calibrated),
    )

    logger.info("--- 2. what else fits this well ---")
    demo_study = collect_behavioural_sets(
        _demo_objective,
        DEMO_BOUNDS,
        calibrated,
        tolerance=DEFAULT_BEHAVIOURAL_TOLERANCE,
        n_starts=DEFAULT_N_STARTS,
    )

    logger.info("--- 3. spread vs trade-off: two different findings ---")
    for name_a, name_b, correlation in compensating_pairs(demo_study):
        logger.info("  compensating: %s <-> %s   r = %+.3f", name_a, name_b, correlation)
    logger.info(
        "  r_internal_ratio roams %.1f%% of its range and correlates with nothing "
        "-- inert, not compensating",
        100 * parameter_spread(demo_study)["r_internal_ratio"].span_fraction,
    )

    logger.info("--- 4. the exhibit: two admissible, opposite buildings ---")
    low_gain, high_gain = most_divergent_pair(demo_study, "internal_gain_w_per_m2")
    for label, entry in (("A", low_gain), ("B", high_gain)):
        logger.info(
            "  set %s (objective %.6f): gain %6.1f W/m2, vent %6.1f kg/s",
            label,
            entry.objective,
            entry.parameters[0],
            entry.parameters[1],
        )

    logger.info("--- 5. what it costs: the same retrofit, priced twice ---")
    saving = outcome_spread(demo_study, _demo_retrofit_saving_pct)
    logger.info("  cut ventilation flow by 30%% -> %s", saving.summary("%"))
    logger.info(
        "  set A says %.1f%%, set B says %.1f%% -- same CV(RMSE), different "
        "business case",
        _demo_retrofit_saving_pct(np.asarray(low_gain.parameters)),
        _demo_retrofit_saving_pct(np.asarray(high_gain.parameters)),
    )

"""The objective function -- deciding what "best fit" actually means.

An optimiser does exactly what it is told. Everything about the
calibrated model -- which parameters end up where, whether the G14 gate
passes, whether the answer is physically sane -- is decided here, not in
L6.7's search algorithm. The search only finds the minimum of whatever
this file defines.

Three decisions are made in this module, and each one changes the answer:

1. **What to minimise.** CV(RMSE) alone is the obvious choice and it is
   wrong for this project: `ashrae_g14_pass()` requires BOTH metrics, so
   an objective blind to NMBE can converge on a fit that scores well on
   the thing being minimised and still fails the gate. See
   `g14_objective`.

2. **How to weight the terms.** NMBE and CV(RMSE) live on different
   scales with different acceptance limits, so `cvrmse + |nmbe|` implies
   an arbitrary exchange rate between them. This module normalises each
   term by its own G14 threshold instead, so both read as "fraction of
   the allowance used" and a value of 1.0 means "exactly at the limit"
   for either. That is a defensible weighting rather than a taste.

3. **How to handle physically impossible parameter sets.** As penalties
   that scale with the size of the violation, never as hard rejections
   -- see `physical_penalty` for why a constant penalty is nearly as bad
   as a crash.

`02_CURRICULUM.md` L6.6 asks for three different objectives to be
compared, and for the disagreement to be shown rather than asserted.
That comparison lives in `notebooks/05_calibration.ipynb`; this module
provides the pieces it compares.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution, minimize

from cooling_twin import SEED
from cooling_twin.calibration.metrics import (
    DataInterval,
    cvrmse,
    g14_thresholds,
    nmbe,
)

logger = logging.getLogger(__name__)

# Returned instead of a non-finite value when a parameter set cannot be
# simulated at all (solver failure). Large enough that no feasible point
# competes with it, finite so that Morris (L6.5) and the optimiser
# (L6.7) can both still form differences. `np.inf` would poison every
# elementary effect it touched; `np.nan` would silently win comparisons
# in some optimisers.
INFEASIBLE_OBJECTIVE = 1.0e6

# Default weight on the physical-penalty term. 10.0 means a violation
# equal to 100% of its allowance costs the same as blowing the entire
# G14 CV(RMSE) budget ten times over -- deliberately steep, because a
# physically impossible fit is not a slightly worse fit, it is a
# different kind of answer.
DEFAULT_PENALTY_WEIGHT = 10.0

# --- two-stage search settings (L6.7) -------------------------------------
#
# Differential evolution cost is roughly popsize * n_params * maxiter
# objective evaluations. For this project one evaluation is a year-long
# ODE solve, so these two numbers are a WALL-CLOCK decision, not a
# statistical one: 15 * 4 * 60 ~ 3,600 simulations. Raise maxiter before
# raising popsize -- a larger population explores more but converges
# slower, and the surface here has few parameters, not many.
DEFAULT_DE_POPSIZE = 15
DEFAULT_DE_MAXITER = 60

# DE stops when the population's spread falls below
# tol * |mean objective|. 0.01 means "the whole population agrees to
# within 1%" -- past that point DE is a bad local optimiser and the
# L-BFGS-B stage should take over.
DEFAULT_DE_TOL = 0.01

# Finite-difference step for the local stage, as a FRACTION of each
# parameter's bound width. SciPy's default `eps` is ~1.5e-8 ABSOLUTE,
# which is meaningless when one parameter spans 0.3-3.0 and another
# spans 5-200: for the second, that step is smaller than the ODE
# solver's own tolerance, so the "gradient" is integrator noise. Scaling
# by bound width makes the step mean the same thing for every parameter.
LOCAL_STEP_FRACTION = 1.0e-4

# A parameter is reported as pinned when it sits within this fraction of
# a bound width from that bound. 1% is tight enough that an interior
# optimum never trips it and loose enough that a bound-seeking optimum
# always does.
PINNED_TOLERANCE = 0.01


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """One objective evaluation, decomposed into the terms that made it.

    Returned instead of a bare float so a calibration run can be
    explained afterwards. "The objective was 1.43" says nothing; "0.61
    of CV(RMSE) budget, 0.82 of NMBE budget, no penalty" says which
    criterion is actually binding and therefore what to change. L6.7's
    artifact log records these, and L6.10's report needs them.

    Attributes:
        cvrmse_pct: Raw CV(RMSE), percent.
        nmbe_pct: Raw NMBE, percent (signed, G14 convention).
        cvrmse_term: `cvrmse_pct / cvrmse_limit` -- fraction of budget.
        nmbe_term: `|nmbe_pct| / nmbe_limit` -- fraction of budget.
        penalty: Physical-violation penalty, already weighted.
        total: What the optimiser minimises.
    """

    cvrmse_pct: float
    nmbe_pct: float
    cvrmse_term: float
    nmbe_term: float
    penalty: float
    total: float

    @property
    def binding_criterion(self) -> str:
        """Which term is currently costing the most.

        The single most useful line in a calibration log: it says what
        the optimiser is actually fighting.
        """
        terms = {
            "penalty": self.penalty,
            "nmbe": self.nmbe_term,
            "cvrmse": self.cvrmse_term,
        }
        return max(terms, key=lambda key: terms[key])

    def summary(self) -> str:
        """One-line human-readable breakdown, for logs."""
        return (
            f"total={self.total:.4f} "
            f"(cvrmse {self.cvrmse_pct:6.2f}% -> {self.cvrmse_term:.3f}, "
            f"nmbe {self.nmbe_pct:+6.2f}% -> {self.nmbe_term:.3f}, "
            f"penalty {self.penalty:.3f}); binding: {self.binding_criterion}"
        )


def physical_penalty(
    violations: Mapping[str, float],
    weight: float = DEFAULT_PENALTY_WEIGHT,
) -> float:
    """Turn physical-invariant violations into an additive penalty.

    Each entry is a violation expressed as a NON-NEGATIVE fraction of
    its own allowance: 0.0 means satisfied, 0.5 means half a budget
    over, 2.0 means three times the allowed amount. Normalising at the
    call site keeps this function from needing to know the units of
    every invariant in the project.

    Why a penalty rather than a hard constraint. The obvious
    alternative is to reject infeasible parameter sets outright --
    raise, or return `INFEASIBLE_OBJECTIVE`. Three things break:

    - Differential evolution (L6.7) mutates and recombines whole
      populations. If every infeasible point returns the same large
      constant, the entire infeasible region is a plateau: the
      optimiser gets no information about WHICH direction is less
      infeasible, and a population initialised largely inside it cannot
      find its way out.
    - The feasible region here is not a box. Box bounds the optimiser
      already handles natively; violations of derived quantities
      (INV-1's COP ceiling, a load that requires the chiller to heat)
      are not expressible as bounds on the parameters themselves.
    - The optimum frequently sits ON a physical boundary. A hard
      rejection makes the best answer unreachable from one side.

    Why the penalty must SCALE with the violation, which is the part
    most implementations get wrong: a constant penalty has zero
    gradient, so it tells the optimiser that being barely infeasible
    and being wildly infeasible are equally bad. Scaling gives the
    infeasible region a slope pointing back toward feasibility.

    Args:
        violations: Mapping of invariant name to normalised violation
            magnitude (0.0 when satisfied).
        weight: Multiplier applied to the summed violations.

    Returns:
        The penalty to add to the objective. 0.0 when nothing is
        violated.

    Raises:
        ValueError: If a violation is negative, non-finite, or if
            `weight` is negative. A negative violation would act as a
            REWARD for breaking physics.
    """
    if weight < 0.0:
        raise ValueError(f"weight must be >= 0, got {weight}")

    total = 0.0
    for name, magnitude in violations.items():
        if not np.isfinite(magnitude):
            raise ValueError(f"violation {name!r} is not finite: {magnitude}")
        if magnitude < 0.0:
            raise ValueError(
                f"violation {name!r} is negative ({magnitude}). Violations are "
                "magnitudes; a negative value would reward the optimiser for "
                "breaking the invariant instead of penalising it."
            )
        total += magnitude

    if total > 0.0:
        logger.debug(
            "physical penalty %.4f from violations: %s",
            weight * total,
            {k: round(v, 4) for k, v in violations.items() if v > 0},
        )
    return weight * total


def clipping_violation(
    raw_load_kw: npt.ArrayLike,
    max_clipped_fraction: float = 0.05,
) -> float:
    """Violation magnitude for a model that only fits by switching off.

    The inverse model clips required cooling at zero, because a chiller
    cannot add heat. That clip is physically correct, but it is also an
    escape hatch the optimiser will exploit: pushing internal gain low
    enough makes the model predict zero for much of the year, which can
    reduce CV(RMSE) on a building whose load never actually reaches
    zero.

    For a laboratory in Arizona (see Q6) a genuinely zero cooling load
    is rare, so a large clipped fraction is evidence of a wrong
    parameter set rather than of a real shoulder season.

    Args:
        raw_load_kw: Required cooling BEFORE clipping, kW. Negative
            entries are the ones that would be clipped.
        max_clipped_fraction: Allowance, as a fraction of all hours.
            Defaults to 5%.

    Returns:
        0.0 if the clipped fraction is within the allowance, otherwise
        the excess expressed as a fraction of that allowance.

    Raises:
        ValueError: If `max_clipped_fraction` is not in (0, 1], or if
            `raw_load_kw` is empty or non-finite.
    """
    if not 0.0 < max_clipped_fraction <= 1.0:
        raise ValueError(
            f"max_clipped_fraction must be in (0, 1], got {max_clipped_fraction}"
        )
    raw = np.asarray(raw_load_kw, dtype=float)
    if raw.size == 0:
        raise ValueError("raw_load_kw must contain at least one point")
    if not np.all(np.isfinite(raw)):
        raise ValueError("raw_load_kw must be finite")

    clipped_fraction = float(np.mean(raw < 0.0))
    excess = clipped_fraction - max_clipped_fraction
    return max(0.0, excess / max_clipped_fraction)


def g14_objective(
    measured: npt.ArrayLike,
    predicted: npt.ArrayLike,
    n_params: int,
    violations: Mapping[str, float] | None = None,
    interval: DataInterval = DataInterval.HOURLY,
    nmbe_weight: float = 1.0,
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT,
) -> ObjectiveBreakdown:
    """The project's calibration objective: both G14 criteria, plus penalties.

    Each metric is divided by its own G14 acceptance limit before the
    two are added, so the sum is in units of "acceptance budget" and
    both terms mean the same thing at the same number. That is what
    makes the weighting defensible: `cvrmse_pct + abs(nmbe_pct)` would
    silently declare 1 point of CV(RMSE) equal to 1 point of NMBE, when
    the standard allows three times as much of the former.

    A `total` below 1.0 does not by itself mean the gate passes -- the
    sum can be under 1 with one term over it. `ashrae_g14_pass()`
    remains the authority on pass/fail; this function only decides what
    the optimiser walks toward.

    Args:
        measured: Measured values `y`.
        predicted: Model output `yhat`, aligned with `measured`.
        n_params: Number of calibrated parameters `p` (L6.2's `n - p`).
        violations: Normalised physical violations, as accepted by
            `physical_penalty`. `None` means nothing was checked, which
            is not the same as nothing being violated.
        interval: Which G14 threshold set normalises the terms.
        nmbe_weight: Relative weight on the NMBE budget term. 1.0 --
            the default -- treats "half the NMBE allowance" and "half
            the CV(RMSE) allowance" as equally costly. Raise it when a
            run keeps landing inside CV(RMSE) but outside NMBE.
        penalty_weight: Passed to `physical_penalty`.

    Returns:
        An `ObjectiveBreakdown`. Minimise `.total`.

    Raises:
        ValueError: If `nmbe_weight` is negative, or for any input
            problem raised by `nmbe`/`cvrmse`/`physical_penalty`.
    """
    if nmbe_weight < 0.0:
        raise ValueError(f"nmbe_weight must be >= 0, got {nmbe_weight}")

    nmbe_limit_pct, cvrmse_limit_pct = g14_thresholds(interval)

    cvrmse_pct = cvrmse(measured, predicted, n_params)
    nmbe_pct = nmbe(measured, predicted, n_params)

    cvrmse_term = cvrmse_pct / cvrmse_limit_pct
    nmbe_term = abs(nmbe_pct) / nmbe_limit_pct
    penalty = physical_penalty(violations or {}, weight=penalty_weight)

    return ObjectiveBreakdown(
        cvrmse_pct=cvrmse_pct,
        nmbe_pct=nmbe_pct,
        cvrmse_term=cvrmse_term,
        nmbe_term=nmbe_term,
        penalty=penalty,
        total=cvrmse_term + nmbe_weight * nmbe_term + penalty,
    )


@dataclass(frozen=True)
class CalibrationResult:
    """Everything needed to reproduce, defend, or reject one calibration run.

    A bare parameter vector is not a result. Six months from now the
    only questions anyone asks are "which data, which objective, which
    seed, and was it actually converged" -- so those travel with the
    numbers, and `write_artifact` puts them on disk.

    Attributes:
        parameter_names: Calibrated parameter names, in vector order.
        best_parameters: The winning vector, same order.
        bounds: The box the search was allowed to use, same order.
        objective_value: Best objective found (the accepted stage's).
        global_objective: Best value at the end of the DE stage.
        local_objective: Value after the L-BFGS-B refinement.
        accepted_stage: `"global"` or `"local"` -- which one is being
            reported. `"global"` means the refinement made things worse
            and was discarded; see `calibrate`.
        n_evaluations: Total objective calls across both stages.
        global_message: SciPy's DE termination message.
        local_message: SciPy's L-BFGS-B termination message.
        pinned_parameters: Names sitting on a bound (see
            `PINNED_TOLERANCE`). Non-empty means the answer is a
            statement about the bounds, not about the building.
        seed: Seed given to DE's sampler.
        elapsed_seconds: Wall clock for the whole run.
        timestamp_utc: ISO-8601, UTC, when the run finished.
        breakdown: Objective decomposition at the winning point, when a
            `breakdown_fn` was supplied.
        metadata: Free-form run context (building, year, data hash).
    """

    parameter_names: tuple[str, ...]
    best_parameters: tuple[float, ...]
    bounds: tuple[tuple[float, float], ...]
    objective_value: float
    global_objective: float
    local_objective: float
    accepted_stage: str
    n_evaluations: int
    global_message: str
    local_message: str
    pinned_parameters: tuple[str, ...]
    seed: int
    elapsed_seconds: float
    timestamp_utc: str
    breakdown: ObjectiveBreakdown | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def parameters(self) -> dict[str, float]:
        """The winning vector as a name -> value mapping."""
        return dict(zip(self.parameter_names, self.best_parameters, strict=True))

    @property
    def local_improvement(self) -> float:
        """How much the local stage bought, as a fraction of the DE value.

        Near zero means DE had already converged and the refinement was
        cosmetic. Large means DE stopped early -- raise `maxiter`
        rather than trusting the polish to rescue it.
        """
        if self.global_objective == 0.0:
            return 0.0
        return (self.global_objective - self.local_objective) / abs(self.global_objective)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the run."""
        record: dict[str, Any] = {
            "timestamp_utc": self.timestamp_utc,
            "seed": self.seed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "objective_value": self.objective_value,
            "accepted_stage": self.accepted_stage,
            "parameters": self.parameters,
            "bounds": {
                name: list(bound)
                for name, bound in zip(self.parameter_names, self.bounds, strict=True)
            },
            "pinned_parameters": list(self.pinned_parameters),
            "stages": {
                "global": {
                    "objective": self.global_objective,
                    "message": self.global_message,
                },
                "local": {
                    "objective": self.local_objective,
                    "message": self.local_message,
                    "relative_improvement": self.local_improvement,
                },
            },
            "n_evaluations": self.n_evaluations,
            "metadata": dict(self.metadata),
        }
        if self.breakdown is not None:
            record["breakdown"] = {
                "cvrmse_pct": self.breakdown.cvrmse_pct,
                "nmbe_pct": self.breakdown.nmbe_pct,
                "cvrmse_term": self.breakdown.cvrmse_term,
                "nmbe_term": self.breakdown.nmbe_term,
                "penalty": self.breakdown.penalty,
                "total": self.breakdown.total,
                "binding_criterion": self.breakdown.binding_criterion,
            }
        return record

    def summary(self) -> str:
        """Multi-line human-readable summary, for logs and notebooks."""
        lines = [
            f"objective {self.objective_value:.4f} "
            f"(global {self.global_objective:.4f} -> local {self.local_objective:.4f}, "
            f"accepted: {self.accepted_stage})",
            f"{self.n_evaluations} evaluations in {self.elapsed_seconds:.1f}s",
        ]
        lines.extend(f"  {name:<28} {value:12.4f}" for name, value in self.parameters.items())
        if self.pinned_parameters:
            lines.append(f"  PINNED AT BOUNDS: {', '.join(self.pinned_parameters)}")
        if self.breakdown is not None:
            lines.append(f"  {self.breakdown.summary()}")
        return "\n".join(lines)


def write_artifact(result: CalibrationResult, directory: Path | str) -> Path:
    """Write one calibration run to a timestamped JSON file.

    Every run writes a record, including the failed ones. A calibration
    you cannot reproduce is not evidence, and "I reran it and got
    something different" is the single most common way a result stops
    being believed. The filename carries the timestamp and seed so runs
    sort chronologically and never overwrite each other.

    Args:
        result: The run to record.
        directory: Destination directory. Created if absent.

    Returns:
        Path to the file written.

    Raises:
        ValueError: If the record cannot be serialised to JSON --
            raised here, loudly, rather than leaving a half-written
            file behind.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    stamp = result.timestamp_utc.replace(":", "").replace("-", "")
    path = target_dir / f"calibration_{stamp}_seed{result.seed}.json"

    try:
        payload = json.dumps(result.to_dict(), indent=2)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"calibration record is not JSON-serialisable: {error}. Check "
            "`metadata` -- numpy scalars and Timestamps need converting to "
            "plain Python types before they go in."
        ) from error

    path.write_text(payload, encoding="utf-8")
    logger.info("calibration artifact written: %s", path)
    return path


def find_pinned_parameters(
    names: tuple[str, ...],
    values: npt.NDArray[np.float64],
    bounds: tuple[tuple[float, float], ...],
    tolerance: float = PINNED_TOLERANCE,
) -> tuple[str, ...]:
    """Names of parameters sitting on a bound.

    `06_ASSESSMENT.md`'s diagnostic table treats this as a finding in
    its own right: an optimum on a bound is not an optimum, it is the
    search being stopped by the box. Either the bound is wrong, or --
    the harder case -- the model has no term for something the fit is
    trying to buy with the parameters it does have.

    Args:
        names: Parameter names, in vector order.
        values: The fitted vector.
        bounds: `(lower, upper)` per parameter, same order.
        tolerance: Distance from a bound, as a fraction of bound width.

    Returns:
        Names of pinned parameters, in vector order.

    Raises:
        ValueError: If `tolerance` is not in (0, 0.5).
    """
    if not 0.0 < tolerance < 0.5:
        raise ValueError(f"tolerance must be in (0, 0.5), got {tolerance}")

    pinned = []
    for name, value, (lower, upper) in zip(names, values, bounds, strict=True):
        margin = tolerance * (upper - lower)
        if value <= lower + margin or value >= upper - margin:
            pinned.append(name)
    return tuple(pinned)


class _FiniteObjective:
    """Picklable wrapper that keeps non-finite values away from the optimiser.

    Not a nested closure, and that is the whole point: `workers != 1`
    hands the objective to a `multiprocessing` pool, which pickles it.
    A closure defined inside `calibrate` cannot be pickled, so wrapping
    the user's objective in one would make parallel search impossible
    no matter how picklable the user's own objective was.

    Attributes:
        objective: The wrapped objective.
    """

    def __init__(self, objective: Callable[[npt.NDArray[np.float64]], float]) -> None:
        self.objective = objective

    def __call__(self, vector: npt.NDArray[np.float64]) -> float:
        """Evaluate, substituting `INFEASIBLE_OBJECTIVE` for NaN/inf."""
        value = float(self.objective(np.asarray(vector, dtype=float)))
        if not np.isfinite(value):
            # Never let a NaN reach the optimiser: `nan < best` is False
            # in DE's comparison but True in some others, so the failure
            # mode is silent and inconsistent between optimisers.
            logger.warning("objective returned %s; substituting INFEASIBLE_OBJECTIVE", value)
            return INFEASIBLE_OBJECTIVE
        return value


def select_better_stage(
    global_x: npt.NDArray[np.float64],
    global_objective: float,
    local_x: npt.NDArray[np.float64],
    local_objective: float,
) -> tuple[npt.NDArray[np.float64], float, str]:
    """Keep the better of the two stages -- the local one is a candidate.

    A gradient method is normally guaranteed not to return worse than
    its starting point, so this looks paranoid. It is not: L-BFGS-B
    estimates its gradient by finite differences, and the objective
    here is an adaptive-step ODE solve, whose step selection changes
    with the parameters. That makes the surface slightly discontinuous
    at the scale of the solver's own tolerance, and a line search over
    a discontinuous function can terminate below its start in the
    solver's internal model while being above it in reality.

    Args:
        global_x: DE's optimum.
        global_objective: Objective there.
        local_x: L-BFGS-B's optimum.
        local_objective: Objective there.

    Returns:
        `(parameters, objective, stage_name)` for the stage kept.
    """
    if local_objective <= global_objective:
        return local_x, local_objective, "local"

    logger.warning(
        "local refinement made the objective WORSE (%.6f -> %.6f) and was "
        "discarded. That is finite-difference noise over an adaptive ODE "
        "solver, not a bug -- but if it happens on every run, raise "
        "LOCAL_STEP_FRACTION so the gradient step clears the solver's own "
        "tolerance.",
        global_objective,
        local_objective,
    )
    return global_x, global_objective, "global"


def calibrate(
    objective: Callable[[npt.NDArray[np.float64]], float],
    bounds: Mapping[str, tuple[float, float]],
    breakdown_fn: Callable[[npt.NDArray[np.float64]], ObjectiveBreakdown] | None = None,
    popsize: int = DEFAULT_DE_POPSIZE,
    maxiter: int = DEFAULT_DE_MAXITER,
    tol: float = DEFAULT_DE_TOL,
    seed: int = SEED,
    workers: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Calibrate by global search first, then local refinement.

    Stage 1 is differential evolution: a population-based, derivative-
    free global search that can cross a ridge between two basins.
    Stage 2 is L-BFGS-B started from DE's answer: a gradient method that
    walks straight downhill and stops when it can no longer improve.

    The order is the point. L-BFGS-B alone finds the bottom of whatever
    basin it starts in -- with a nominal start on a multi-modal surface
    that is a coin toss dressed as an answer. DE alone gets into the
    right basin quickly but converges slowly inside it, spending
    thousands of evaluations to buy the last percent. Global then local
    takes the strength of each.

    The local stage is only accepted if it IMPROVES on DE. Finite
    differences over an adaptive-step ODE solver are noisy -- the
    solver's own step selection changes with the parameters, so the
    objective has small discontinuities -- and a gradient method fed a
    noisy gradient can land worse than where it started. Silently
    reporting that is how a calibration quietly degrades.

    Args:
        objective: Maps a parameter vector, ordered as `bounds`, to the
            scalar being minimised. Must be deterministic and finite;
            return `INFEASIBLE_OBJECTIVE` for parameter sets that
            cannot be simulated.
        bounds: Ordered mapping of parameter name to `(lower, upper)`.
            Insertion order defines the vector order everywhere else.
        breakdown_fn: Optional second evaluation at the winning point,
            returning an `ObjectiveBreakdown` for the artifact log.
            Costs one extra simulation and makes the record explain
            itself.
        popsize: DE population multiplier (population is
            `popsize * n_params`).
        maxiter: DE generation cap.
        tol: DE convergence tolerance.
        seed: Passed to DE's sampler. Two runs with the same seed and
            the same data must produce the same answer (L0.3).
        workers: Passed to DE. `-1` uses every core. The objective
            must then be picklable: a closure or a lambda (including
            one defined in a notebook cell) is not, and SciPy reports
            it as an opaque pickling error from inside its pool. A
            module-level function, or an instance of a class with
            `__call__`, both work.
        metadata: Run context to record in the artifact (building, year,
            data fingerprint, objective name).

    Returns:
        A `CalibrationResult`.

    Raises:
        ValueError: If `bounds` is empty, if any bound is inverted or
            non-finite, or if the objective returns a non-finite value
            at the DE optimum.
    """
    if not bounds:
        raise ValueError("bounds must contain at least one parameter")

    names = tuple(bounds)
    bound_pairs = tuple((float(low), float(high)) for low, high in bounds.values())
    for name, (low, high) in zip(names, bound_pairs, strict=True):
        if not np.isfinite([low, high]).all():
            raise ValueError(f"{name}: bounds must be finite, got ({low}, {high})")
        if not low < high:
            raise ValueError(
                f"{name}: lower bound ({low}) must be strictly less than upper "
                f"({high}). A collapsed bound silently removes the parameter "
                "from the calibration while still charging a degree of freedom "
                "for it in n - p."
            )

    guarded = _FiniteObjective(objective)
    started = time.perf_counter()

    logger.info(
        "stage 1/2 differential evolution: %d parameters, popsize %d, maxiter %d",
        len(names),
        popsize,
        maxiter,
    )
    global_result = differential_evolution(
        guarded,
        bounds=list(bound_pairs),
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        seed=seed,
        # DE's own `polish=True` runs L-BFGS-B with SciPy's default
        # settings and folds the result in silently. We do that stage
        # ourselves so it is separately reported, uses a parameter-scaled
        # finite-difference step, and can be REJECTED when it makes
        # things worse.
        polish=False,
        init="latinhypercube",
        updating="immediate" if workers == 1 else "deferred",
        workers=workers,
    )
    global_x = np.asarray(global_result.x, dtype=float)
    global_objective = float(global_result.fun)
    logger.info(
        "stage 1/2 done: objective %.6f after %d evaluations (%s)",
        global_objective,
        int(global_result.nfev),
        global_result.message,
    )

    # No finiteness check is needed here: `guarded` is the only thing DE
    # ever evaluates and it cannot return a non-finite value.

    # Parameter-scaled finite-difference step (see LOCAL_STEP_FRACTION).
    steps = np.array([LOCAL_STEP_FRACTION * (high - low) for low, high in bound_pairs])

    logger.info("stage 2/2 L-BFGS-B refinement from the DE optimum")
    local_result = minimize(
        guarded,
        x0=global_x,
        method="L-BFGS-B",
        bounds=bound_pairs,
        options={"eps": steps, "maxiter": 200},
    )
    local_x = np.asarray(local_result.x, dtype=float)
    local_objective = float(local_result.fun)

    best_x, best_objective, accepted_stage = select_better_stage(
        global_x, global_objective, local_x, local_objective
    )

    elapsed = time.perf_counter() - started
    pinned = find_pinned_parameters(names, best_x, bound_pairs)
    if pinned:
        logger.warning(
            "parameters pinned at their bounds: %s. Treat this as a finding, "
            "not a result: either the bound is wrong, or the model has no term "
            "for something the fit is trying to buy with these parameters.",
            ", ".join(pinned),
        )

    breakdown = breakdown_fn(best_x) if breakdown_fn is not None else None

    result = CalibrationResult(
        parameter_names=names,
        best_parameters=tuple(float(value) for value in best_x),
        bounds=bound_pairs,
        objective_value=best_objective,
        global_objective=global_objective,
        local_objective=local_objective,
        accepted_stage=accepted_stage,
        # SciPy's own counts, not a hand-rolled one: a worker process
        # cannot increment a counter in the parent, so counting locally
        # would silently under-report every parallel run.
        n_evaluations=int(global_result.nfev) + int(local_result.nfev),
        global_message=str(global_result.message),
        local_message=str(local_result.message),
        pinned_parameters=pinned,
        seed=seed,
        elapsed_seconds=elapsed,
        timestamp_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        breakdown=breakdown,
        metadata=dict(metadata or {}),
    )
    logger.info("calibration finished\n%s", result.summary())
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    measured_demo = np.array([100.0, 200.0, 300.0, 400.0])

    logger.info("--- 1. what the budget normalisation does ---")
    # Model A: CV(RMSE) is good, NMBE is bad (a uniform 8% under-prediction
    # plus small scatter). Model B: the reverse.
    model_a = measured_demo * 0.92
    model_b = measured_demo + np.array([-60.0, 60.0, -60.0, 60.0])

    for label, model in (("A: biased, tight", model_a), ("B: unbiased, loose", model_b)):
        breakdown = g14_objective(measured_demo, model, n_params=0)
        logger.info("%-20s %s", label, breakdown.summary())

    logger.info("--- 2. a naive objective disagrees ---")
    for label, model in (("A: biased, tight", model_a), ("B: unbiased, loose", model_b)):
        naive = cvrmse(measured_demo, model, n_params=0) + abs(
            nmbe(measured_demo, model, n_params=0)
        )
        logger.info("%-20s cvrmse + |nmbe| = %.4f", label, naive)

    logger.info("--- 3. penalties scale with the violation ---")
    # A year of required cooling that mostly stays positive. Lowering the
    # assumed internal gain pushes more hours below zero, i.e. makes the
    # model "switch off" to fit -- the escape hatch the penalty closes.
    raw_load_kw = np.linspace(50.0, 2000.0, 8760)
    for assumed_gain_kw in (0.0, -100.0, -200.0, -400.0, -800.0):
        shifted = raw_load_kw + assumed_gain_kw
        violation = clipping_violation(shifted, max_clipped_fraction=0.05)
        breakdown = g14_objective(
            measured_demo,
            model_a,
            n_params=0,
            violations={"clipped_hours": violation},
        )
        logger.info(
            "gain shift %+7.0f kW -> clipped %5.1f%%  violation %5.2f  total %8.4f  binding: %s",
            assumed_gain_kw,
            100 * float(np.mean(shifted < 0)),
            violation,
            breakdown.total,
            breakdown.binding_criterion,
        )

    logger.info("--- 4. why global BEFORE local ---")
    # A known-answer surface with the two features that break a local
    # optimiser on this project's real objective: a broad bowl (the fit
    # gets better as total load approaches the measured total) with
    # ripples on it (the ODE solver's step selection and the clipping
    # threshold both make the surface locally bumpy). Minimum is exactly
    # 0.0 at TRUTH, and every ripple trough is a false answer that looks
    # converged.
    DEMO_BOUNDS = {
        "internal_gain_w_per_m2": (5.0, 200.0),
        "ua_envelope_w_per_m2k": (0.3, 3.0),
    }
    TRUTH = np.array([120.0, 1.7])
    RIPPLE_AMPLITUDE = 0.15
    RIPPLES_PER_RANGE = 4.0

    demo_lower = np.array([low for low, _ in DEMO_BOUNDS.values()])
    demo_width = np.array([high - low for low, high in DEMO_BOUNDS.values()])

    def rippled_bowl(vector: npt.NDArray[np.float64]) -> float:
        """Known-answer objective: 0.0 at TRUTH, local minima everywhere else."""
        offset = (np.asarray(vector, dtype=float) - TRUTH) / demo_width
        bowl = float(np.sum(offset**2))
        ripple = RIPPLE_AMPLITUDE * float(
            np.sum(1.0 - np.cos(2.0 * np.pi * RIPPLES_PER_RANGE * offset))
        )
        return bowl + ripple

    # The local-only strategy, started from L6.1's honest engineering
    # guess (15 W/m2 internal gain, UA 1.0) -- which is exactly what
    # "start the optimiser from the nominal values" means here, since
    # there is no data sheet for this building.
    nominal_start = np.array([15.0, 1.0])
    local_only = minimize(
        rippled_bowl,
        x0=nominal_start,
        method="L-BFGS-B",
        bounds=list(DEMO_BOUNDS.values()),
        options={"eps": LOCAL_STEP_FRACTION * demo_width},
    )
    logger.info(
        "local only  : objective %.6f at gain %.1f, ua %.3f  (truth: %.1f, %.3f)",
        float(local_only.fun),
        local_only.x[0],
        local_only.x[1],
        TRUTH[0],
        TRUTH[1],
    )

    two_stage = calibrate(
        rippled_bowl,
        DEMO_BOUNDS,
        metadata={"demo": "rippled_bowl", "truth": TRUTH.tolist()},
    )
    logger.info(
        "two stage   : objective %.6f at gain %.1f, ua %.3f  (%d evaluations)",
        two_stage.objective_value,
        two_stage.parameters["internal_gain_w_per_m2"],
        two_stage.parameters["ua_envelope_w_per_m2k"],
        two_stage.n_evaluations,
    )

    logger.info("--- 5. the artifact ---")
    with tempfile.TemporaryDirectory() as demo_dir:
        artifact_path = write_artifact(two_stage, demo_dir)
        logger.info("%s", artifact_path.read_text(encoding="utf-8"))

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

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

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

"""Morris screening -- deciding WHICH parameters are worth calibrating.

The RC model plus the plant models expose more adjustable numbers than
should ever be fitted at once. `06_ASSESSMENT.md`'s M6 gate caps the
calibration at 10 parameters, and there is a real reason for the cap
beyond tidiness: every fitted parameter costs a degree of freedom in
`n - p` (L6.2), widens the space the optimiser must search (L6.7), and
adds one more axis along which two different parameter sets can produce
the same output (equifinality, L6.8 -- already seen empirically in
L4.4's 4R3C restarts).

So the question this module answers is not "what are the best parameter
values" but "which parameters is it even worth trying to fit". Morris
screening answers exactly that, cheaply:

    cost:      N * (k + 1) model runs, for k parameters
    output:    mu*  -- how much this parameter moves the objective
               sigma -- how much that effect depends on the others

A parameter with low mu* can be fixed at its nominal value and removed
from the calibration entirely. A parameter with high sigma cannot be
reasoned about in isolation, which is a warning about the optimiser's
job, not about the parameter.

Why Morris rather than Sobol: Sobol gives a variance decomposition --
strictly more information -- but needs roughly `N * (2k + 2)` runs with
N in the thousands to converge, versus Morris's `N * (k + 1)` with N in
the tens. For a screening question ("which of these 14 matter?") the
extra precision buys nothing: the answer is an ordering and a cut, not
a set of variance fractions. Sobol becomes the right tool once the
parameter set is already small and the question has changed to "how
much of the output variance does each one explain" -- which is an M7
residual-analysis question, not an M6 screening one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from SALib.analyze import morris as morris_analyze
from SALib.sample import morris as morris_sample

from cooling_twin import SEED

logger = logging.getLogger(__name__)

# 06_ASSESSMENT.md M6 supporting requirements: "<= 10 parameters
# calibrated". A hard cap, so it lives in code (the L6.3 rule).
MAX_CALIBRATED_PARAMS = 10

# 02_CURRICULUM.md L6.5 selects the top 8, deliberately below the cap:
# leaving two slots free means a parameter can be added later on
# physical grounds (a fault term in M7, say) without immediately
# breaching the gate.
DEFAULT_TOP_K = 8


@dataclass(frozen=True)
class ParameterSpec:
    """One calibratable parameter and the range it is screened over.

    The range is a PHYSICAL statement, not a search convenience. Morris
    measures how much the objective moves as a parameter sweeps its
    range, so a range set too narrow makes a genuinely important
    parameter look unimportant -- and that artifact is invisible in the
    output. Bounds must come from `03_DOMAIN_REFERENCE.md`, a data
    sheet, or an explicit stated assumption.

    Attributes:
        name: Parameter name, matching the model's own naming.
        lower: Lower bound, inclusive.
        upper: Upper bound, inclusive.
        nominal: The value this parameter is fixed at if screening
            rules it out. Must lie inside the bounds -- a nominal value
            outside the screened range means the screening never
            examined the value actually used.
    """

    name: str
    lower: float
    upper: float
    nominal: float

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError(
                f"{self.name}: lower ({self.lower}) must be strictly less than "
                f"upper ({self.upper}). A zero-width range would make Morris "
                "report a zero elementary effect for a parameter that may "
                "matter a great deal."
            )
        if not self.lower <= self.nominal <= self.upper:
            raise ValueError(
                f"{self.name}: nominal ({self.nominal}) must lie within "
                f"[{self.lower}, {self.upper}] -- otherwise fixing this "
                "parameter at nominal uses a value the screening never tested."
            )


@dataclass(frozen=True)
class MorrisResult:
    """Elementary-effect statistics for every screened parameter.

    Attributes:
        names: Parameter names, in the order they were screened.
        mu_star: Mean of the ABSOLUTE elementary effects. The influence
            ranking. See the L6.5 rationale for why this is not `mu`.
        sigma: Standard deviation of the (signed) elementary effects.
            Large sigma means the parameter's effect depends on where
            the other parameters are -- interaction or non-linearity.
        mu_star_conf: Bootstrap confidence interval half-width on
            `mu_star`. Two parameters whose intervals overlap are not
            distinguishable at this trajectory count.
        n_model_runs: How many model evaluations this cost.
    """

    names: tuple[str, ...]
    mu_star: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]
    mu_star_conf: npt.NDArray[np.float64]
    n_model_runs: int

    def to_frame(self) -> pd.DataFrame:
        """Return the results as a DataFrame ranked by `mu_star`, descending."""
        frame = pd.DataFrame(
            {
                "parameter": self.names,
                "mu_star": self.mu_star,
                "mu_star_conf": self.mu_star_conf,
                "sigma": self.sigma,
            }
        )
        frame["sigma_over_mu_star"] = np.where(
            frame["mu_star"] > 0, frame["sigma"] / frame["mu_star"], np.nan
        )
        return frame.sort_values("mu_star", ascending=False).reset_index(drop=True)


def morris_screening(
    specs: Sequence[ParameterSpec],
    objective: Callable[[npt.NDArray[np.float64]], float],
    n_trajectories: int = 20,
    num_levels: int = 4,
    seed: int = SEED,
) -> MorrisResult:
    """Screen parameters by elementary effects (Morris, 1991).

    Args:
        specs: The parameters to screen, with physical bounds.
        objective: Maps a parameter vector -- ordered exactly as
            `specs` -- to a scalar to be screened, typically CV(RMSE).
            Must be deterministic: Morris attributes every change in
            the output to the one parameter that moved, so a stochastic
            objective is reported as parameter influence.
        n_trajectories: Number of Morris trajectories `N`. Total cost is
            `N * (k + 1)` model runs. 20 is a common screening default;
            check `mu_star_conf` and raise it if the intervals of
            parameters near the cut-off overlap.
        num_levels: Grid levels per parameter. 4 is the standard choice
            and pairs with the `delta = num_levels / (2 * (num_levels - 1))`
            step SALib uses.
        seed: Passed to SALib's sampler so the trajectories -- and
            therefore the whole screening -- are reproducible (L0.3).

    Returns:
        A `MorrisResult`.

    Raises:
        ValueError: If fewer than two parameters are given, if names are
            duplicated, if `n_trajectories` or `num_levels` are out of
            range, or if the objective returns a non-finite value.
    """
    if len(specs) < 2:
        raise ValueError(
            f"Morris screening needs at least 2 parameters, got {len(specs)}. "
            "With one parameter there is nothing to rank."
        )
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"parameter names must be unique, got {names}")
    if n_trajectories < 2:
        raise ValueError(f"n_trajectories must be >= 2, got {n_trajectories}")
    if num_levels < 2:
        raise ValueError(f"num_levels must be >= 2, got {num_levels}")

    problem = {
        "num_vars": len(specs),
        "names": names,
        "bounds": [[spec.lower, spec.upper] for spec in specs],
    }

    samples = morris_sample.sample(
        problem, N=n_trajectories, num_levels=num_levels, seed=seed
    )
    logger.info(
        "Morris screening: %d parameters, %d trajectories -> %d model runs",
        len(specs),
        n_trajectories,
        samples.shape[0],
    )

    outputs = np.empty(samples.shape[0], dtype=float)
    for row_index, parameter_vector in enumerate(samples):
        value = float(objective(parameter_vector))
        if not np.isfinite(value):
            raise ValueError(
                f"objective returned {value} at sample row {row_index} "
                f"({dict(zip(names, parameter_vector, strict=True))}). Morris "
                "cannot form an elementary effect from a non-finite output -- "
                "give the objective an explicit penalty value for infeasible "
                "parameter sets instead."
            )
        outputs[row_index] = value

    analysis = morris_analyze.analyze(
        problem, samples, outputs, num_levels=num_levels, seed=seed
    )

    return MorrisResult(
        # SALib returns numpy str_ objects; cast to plain str so the
        # names compare and print like the ones the caller passed in.
        names=tuple(str(name) for name in analysis["names"]),
        mu_star=np.asarray(analysis["mu_star"], dtype=float),
        sigma=np.asarray(analysis["sigma"], dtype=float),
        mu_star_conf=np.asarray(analysis["mu_star_conf"], dtype=float),
        n_model_runs=int(samples.shape[0]),
    )


def select_top_parameters(
    result: MorrisResult,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[str, ...]:
    """Take the `top_k` most influential parameters, by `mu_star`.

    Everything not returned here is fixed at its `ParameterSpec.nominal`
    for the rest of M6. That is a real decision with a real cost: a
    parameter fixed at a wrong nominal value becomes a permanent bias
    the optimiser cannot remove, and it will show up in M7's residuals
    rather than in the calibration. The trade is accepted because
    fitting a parameter the objective barely responds to is worse --
    the optimiser will move it anywhere at all, consuming a degree of
    freedom in `n - p` and buying nothing.

    Args:
        result: Output of `morris_screening`.
        top_k: How many to keep. Defaults to 8, below
            `MAX_CALIBRATED_PARAMS` so later additions do not breach
            the M6 gate.

    Returns:
        Parameter names, most influential first.

    Raises:
        ValueError: If `top_k` is not positive or exceeds
            `MAX_CALIBRATED_PARAMS`.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_k > MAX_CALIBRATED_PARAMS:
        raise ValueError(
            f"top_k ({top_k}) exceeds MAX_CALIBRATED_PARAMS "
            f"({MAX_CALIBRATED_PARAMS}), the cap set by 06_ASSESSMENT.md's M6 "
            "gate. Raising the cap is a gate change, not a parameter choice."
        )

    ranked = result.to_frame()
    selected = tuple(str(name) for name in ranked["parameter"].head(top_k))

    if top_k < len(ranked):
        cut_mu_star = float(ranked["mu_star"].iloc[top_k - 1])
        next_mu_star = float(ranked["mu_star"].iloc[top_k])
        cut_conf = float(ranked["mu_star_conf"].iloc[top_k - 1])
        if next_mu_star + cut_conf >= cut_mu_star:
            logger.warning(
                "The cut between rank %d (%s, mu*=%.4g) and rank %d (%s, "
                "mu*=%.4g) is within the confidence interval (+/-%.4g) -- "
                "these two are not distinguishable at this trajectory count. "
                "Raise n_trajectories before treating the excluded one as "
                "unimportant.",
                top_k,
                ranked["parameter"].iloc[top_k - 1],
                cut_mu_star,
                top_k + 1,
                ranked["parameter"].iloc[top_k],
                next_mu_star,
                cut_conf,
            )

    return selected


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # A known-answer objective: y depends strongly on a, weakly on b,
    # not at all on c, and on d only through an interaction with a.
    # Morris must recover that ordering, and must give d a large sigma.
    demo_specs = [
        ParameterSpec("a_strong", 0.0, 1.0, nominal=0.5),
        ParameterSpec("b_weak", 0.0, 1.0, nominal=0.5),
        ParameterSpec("c_inert", 0.0, 1.0, nominal=0.5),
        ParameterSpec("d_interacting", 0.0, 1.0, nominal=0.5),
    ]

    def demo_objective(x: npt.NDArray[np.float64]) -> float:
        a, b, _c, d = x
        return float(10.0 * a + 1.0 * b + 8.0 * a * d)

    result = morris_screening(demo_specs, demo_objective, n_trajectories=40)

    logger.info("--- Morris screening on a known-answer objective ---")
    logger.info("cost: %d model runs", result.n_model_runs)
    logger.info("\n%s", result.to_frame().round(4).to_string(index=False))

    logger.info("--- top-2 selection ---")
    logger.info("keep: %s", select_top_parameters(result, top_k=2))
    logger.info(
        "fix at nominal: %s",
        tuple(s.name for s in demo_specs if s.name not in select_top_parameters(result, top_k=2)),
    )

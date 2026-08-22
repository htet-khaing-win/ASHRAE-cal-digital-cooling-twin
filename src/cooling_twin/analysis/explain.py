"""Feature attribution, and what it does not tell you (L7.4).

L7.3 produced the project's interpretability claim as three numbers that
sum to 100: physics %, ML %, unexplained %. This module produces the
OTHER kind of explanation -- the one every reviewer asks for by name --
so the two can be put side by side on the SAME fitted model and the
difference can be shown rather than asserted.

    variance decomposition   how much of the BUILDING the model got
    feature attribution      how the MODEL divides its own output

Those answer different questions, and only the first one can be wrong in
a way the second one reveals. A model that learnt nothing transferable
still yields a confident-looking attribution ranking, because the
ranking is computed from the model's own behaviour and never compares it
to the measured load. `_demo` fits exactly that model -- the same
ensemble on a SHUFFLED residual -- and shows the ranking survive.

WHY EXACT SHAPLEY VALUES RATHER THAN THE `shap` PACKAGE. Shapley values
are exponential in the feature count: 2^k coalitions. With k = 6
(`hybrid.FEATURE_NAMES`) that is 64, which is cheap, so the approximation
the package exists to provide buys nothing here and costs a dependency,
a version pin, and a black box in the middle of the deliverable. The
definition is 30 lines. Computing it exactly also makes the EFFICIENCY
axiom -- the attributions sum to the prediction minus the base value --
a testable invariant rather than a claim in someone's README; see
`shapley_values_kw`, which raises when it is violated.

WHAT A SHAPLEY VALUE ACTUALLY IS, since the plots are usually shown
without it. Take one hour. Ask what the model predicts when it is told
nothing (the average prediction over a background sample, the BASE
VALUE), and what it predicts when told everything (this hour's actual
prediction). The difference has to be divided among the k features. A
Shapley value gives feature j the average, over every possible ORDER of
revealing the features, of how much the prediction moved when j was
revealed. "Averaged over every order" is what makes it unique, and it is
also what makes it expensive.

THE COST NOBODY MENTIONS. To ask "what does the model predict knowing
only outdoor temperature", the temperature of the hour being explained
is pasted onto background rows and everything else is left as it was.
That fabricates inputs: this building's 24-hour mean temperature tracks
its instantaneous temperature closely, and the coalitions cheerfully ask
the model about a 38 degC hour inside a 4 degC week. The model answers,
because a tree ensemble always answers. `ShapleyExplanation` carries
`off_manifold_fraction` so the size of that fabrication is reported
next to the attribution it produced, instead of being left as a
footnote.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import factorial
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from cooling_twin import SEED

logger = logging.getLogger(__name__)

# Above this many features the exact computation stops being defensible
# as a lesson OR as engineering: 2^k coalitions, and the caller should be
# reaching for a sampled estimator (which is what `shap` is) and saying
# so in the report. Refusing loudly is better than silently taking
# twenty minutes.
MAX_EXACT_FEATURES = 12

# Hard ceiling on `2^k * n_explain * n_background` predictions. Sized so
# the default demo (64 x 150 x 120) sits an order of magnitude below it
# and a careless call cannot peg the CPU for an hour -- see CLAUDE.md
# section 10 on this machine.
MAX_MODEL_EVALUATIONS = 8_000_000

# The attributions must sum to `prediction - base_value` exactly; this
# allows only for float64 accumulation over the coalitions.
#
# BE PRECISE ABOUT WHAT THIS CATCHES. Efficiency is an algebraic identity
# of the coalition weights, so it holds for ANY set function v -- it
# cannot notice that the model is nonsense, and it cannot notice a
# non-deterministic `predict`, because each coalition's value is computed
# once and then only differenced. What it catches is a wrong weight, a
# wrong mask, or a mis-indexed marginal contribution: exactly the bugs a
# hand-rolled Shapley implementation is prone to, which is why hand-
# rolling one is defensible here at all.
EFFICIENCY_TOLERANCE_KW = 1e-6

# Below this the base value is an average over too few rows to stand for
# "what the model predicts knowing nothing", and every attribution is
# measured against a lottery. Warned rather than raised: known-answer
# tests legitimately use a handful of rows.
MIN_BACKGROUND_ROWS = 30

# Feature pairs whose joint distribution is physically tight, so that
# mixing one from the explained hour with the other from a background
# hour fabricates weather. Instantaneous dry bulb and its own 24-hour
# mean is the obvious one on `hybrid.FEATURE_NAMES`; the hour harmonics
# are the other (sin and cos of the same angle lie on a circle, and a
# coalition takes one from each of two different hours, landing off it).
DEFAULT_PLAUSIBILITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("outdoor_dry_bulb_c", "outdoor_dry_bulb_24h_mean_c"),
    ("hour_sin", "hour_cos"),
)

_PAIR_SEPARATOR = " vs "


@dataclass(frozen=True, eq=False)
class ShapleyExplanation:
    """Exact Shapley attributions for one fitted model on chosen hours.

    Attributes:
        label: What was explained.
        feature_names: Columns, in the order the values are stored.
        values_kw: `(n_explain, k)` attributions, kW. Signed: a negative
            value means that feature pushed the prediction BELOW the
            base value on that hour.
        base_value_kw: The model's mean prediction over the background
            rows -- what it says knowing nothing.
        prediction_kw: The model's prediction on each explained hour.
        n_background: Background rows the base value averages over.
        efficiency_error_kw: Largest violation of
            `sum(values) == prediction - base_value` across the
            explained hours. Reported, not hidden: it is this
            implementation's self-check on its own coalition arithmetic,
            and it says nothing about whether the model is any good.
        off_manifold_fraction: Share of the fabricated coalition rows
            that fall outside the observed joint range, per feature pair
            (see `DEFAULT_PLAUSIBILITY_PAIRS`). These attributions are
            an average over model behaviour that includes those rows.
    """

    label: str
    feature_names: tuple[str, ...]
    values_kw: npt.NDArray[np.float64]
    base_value_kw: float
    prediction_kw: npt.NDArray[np.float64]
    n_background: int
    efficiency_error_kw: float
    off_manifold_fraction: Mapping[str, float]

    @property
    def mean_abs_kw(self) -> Mapping[str, float]:
        """Mean |attribution| per feature, kW, descending.

        This is the bar chart everyone has seen. The absolute value is
        why it cannot be read as a direction: a feature that adds 300 kW
        in August and removes 300 kW in January ranks identically to one
        that adds 300 kW all year.
        """
        magnitudes = np.abs(self.values_kw).mean(axis=0)
        ranked = sorted(
            zip(self.feature_names, magnitudes, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return MappingProxyType({name: float(value) for name, value in ranked})

    @property
    def ranking(self) -> tuple[str, ...]:
        """Feature names, most attributed first."""
        return tuple(self.mean_abs_kw)

    @property
    def worst_off_manifold_fraction(self) -> float:
        """The largest fabrication rate over the measured pairs."""
        return max(self.off_manifold_fraction.values(), default=0.0)

    def to_frame(self) -> pd.DataFrame:
        """Per-hour attributions, one column per feature."""
        return pd.DataFrame(self.values_kw, columns=list(self.feature_names))


def sample_rows(frame: pd.DataFrame, n_rows: int, *, seed: int = SEED) -> pd.DataFrame:
    """Draw rows without replacement, keeping them in their original order.

    Sampling is the caller's decision and is therefore explicit rather
    than hidden inside `shapley_values_kw`: the cost is quadratic in
    these two counts, and a function that quietly subsampled would make
    the reported attribution depend on a number nobody chose.

    Args:
        frame: Rows to sample from.
        n_rows: How many to draw. Returns the frame itself if it holds
            no more than this.
        seed: Seeds the draw.

    Returns:
        The sampled rows, in the input's own order.

    Raises:
        ValueError: If `n_rows` is not positive or the frame is empty.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if frame.empty:
        raise ValueError("frame is empty")
    if len(frame) <= n_rows:
        return frame
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), size=n_rows, replace=False))
    return frame.iloc[positions]


def _off_manifold_fraction(
    explain: pd.DataFrame,
    background: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
) -> Mapping[str, float]:
    """Share of all coalition rows that fabricate an unobserved pair.

    Counted in closed form rather than by materialising the coalitions.
    For a pair `(a, b)`, a coalition row is fabricated only when exactly
    one of the two columns is taken from the explained hour -- otherwise
    both come from the same real row. Exactly half the 2^k coalitions
    split the pair that way, and each of those halves contributes the
    same n_explain x n_background grid of differences, so the fraction
    over ALL coalition rows is `(out_a + out_b) / (4 * n_e * n_b)` and
    does not depend on k at all.

    "Fabricated" means the difference `a - b` falls outside the range of
    that same difference in the background rows -- a range read from the
    data, not a threshold anyone picked. A NaN difference is counted as
    plausible: unknown is not the same as impossible.
    """
    fractions: dict[str, float] = {}
    for name_a, name_b in pairs:
        observed = (background[name_a] - background[name_b]).to_numpy(dtype=float)
        low, high = float(np.nanmin(observed)), float(np.nanmax(observed))
        column_a = explain[name_a].to_numpy(dtype=float)[:, None]
        column_b = explain[name_b].to_numpy(dtype=float)[:, None]
        with np.errstate(invalid="ignore"):
            mixed_a = column_a - background[name_b].to_numpy(dtype=float)[None, :]
            mixed_b = background[name_a].to_numpy(dtype=float)[None, :] - column_b
            outside = int(
                np.count_nonzero((mixed_a < low) | (mixed_a > high))
                + np.count_nonzero((mixed_b < low) | (mixed_b > high))
            )
        fractions[f"{name_a}{_PAIR_SEPARATOR}{name_b}"] = outside / (
            4.0 * len(explain) * len(background)
        )
    return MappingProxyType(fractions)


def shapley_values_kw(
    model: Any,
    explain: pd.DataFrame,
    background: pd.DataFrame,
    *,
    label: str = "",
    plausibility_pairs: Sequence[tuple[str, str]] | None = None,
) -> ShapleyExplanation:
    """Exact interventional Shapley values, by enumerating every coalition.

    For each subset `S` of features the model is asked what it predicts
    when only `S` is known: the explained hour's values in `S`, every
    background row's values outside `S`, averaged. Feature `j` then
    receives the weighted average of `v(S + j) - v(S)` over all `S` not
    containing `j`, with the weights that make the result unique.

    INTERVENTIONAL, not conditional: features outside `S` are replaced by
    background values independently, rather than sampled from their
    distribution given `S`. The choice matters and is not a detail. The
    interventional form attributes strictly to what the MODEL uses, so a
    feature the model ignores scores exactly zero even when it correlates
    perfectly with one the model uses -- which is the property that makes
    this a statement about the model. The price is the fabricated rows
    counted in `off_manifold_fraction`. The conditional form would keep
    the rows realistic but spread credit onto features the model never
    reads, which is worse here: this module exists to show what the
    correction depends on.

    Args:
        model: A fitted estimator with `predict(X) -> array`. Must be
            deterministic: every number here is an average of its
            outputs, and NOTHING in this function will notice if it is
            not -- see the note on `EFFICIENCY_TOLERANCE_KW`.
        explain: Hours to explain. Columns define the feature order.
        background: Rows standing in for "knowing nothing". Use the
            model's TRAINING distribution: the base value is what the
            model says on average there, and a background drawn from
            somewhere else silently redefines every attribution.
        label: For logs.
        plausibility_pairs: Pairs to measure fabrication on. `None`
            selects those of `DEFAULT_PLAUSIBILITY_PAIRS` present in the
            frame; an explicit pair naming an absent column raises.

    Returns:
        The explanation.

    Raises:
        ValueError: If the frames disagree on columns, are empty, carry
            more than `MAX_EXACT_FEATURES` columns, would exceed
            `MAX_MODEL_EVALUATIONS`, name an absent plausibility column,
            or if the efficiency axiom is violated.
    """
    if list(explain.columns) != list(background.columns):
        raise ValueError(
            "explain and background must carry the same columns in the same order; "
            f"got {list(explain.columns)} and {list(background.columns)}"
        )
    if explain.empty or background.empty:
        raise ValueError("explain and background must both be non-empty")

    names = tuple(str(name) for name in explain.columns)
    n_features = len(names)
    if n_features < 2:
        raise ValueError(
            f"{n_features} feature(s): an attribution over one feature is the "
            "prediction itself and says nothing"
        )
    if n_features > MAX_EXACT_FEATURES:
        raise ValueError(
            f"{n_features} features means 2^{n_features} coalitions, above the "
            f"exact limit of {MAX_EXACT_FEATURES}. Use a sampled estimator and "
            "say in the report that the values are approximate."
        )

    n_explain, n_background = len(explain), len(background)
    evaluations = (1 << n_features) * n_explain * n_background
    if evaluations > MAX_MODEL_EVALUATIONS:
        raise ValueError(
            f"2^{n_features} x {n_explain} x {n_background} = {evaluations:,} model "
            f"evaluations, above the ceiling of {MAX_MODEL_EVALUATIONS:,}. Subsample "
            "with sample_rows() -- the cost is linear in each count and the "
            "attribution is an average, so it converges long before the full frame."
        )
    if n_background < MIN_BACKGROUND_ROWS:
        logger.warning(
            "%s: base value averaged over %d background rows (below %d). Every "
            "attribution is measured against it, so it is a lottery at this size.",
            label or "shapley",
            n_background,
            MIN_BACKGROUND_ROWS,
        )

    if plausibility_pairs is None:
        pairs = tuple(
            pair
            for pair in DEFAULT_PLAUSIBILITY_PAIRS
            if pair[0] in explain.columns and pair[1] in explain.columns
        )
    else:
        pairs = tuple(plausibility_pairs)
        for pair in pairs:
            for name in pair:
                if name not in explain.columns:
                    raise ValueError(f"plausibility pair names absent column {name!r}")

    matrix_explain = explain.to_numpy(dtype=float)
    matrix_background = background.to_numpy(dtype=float)

    # v[mask] holds, for every explained hour, the model's mean
    # prediction when exactly the features in `mask` are revealed. Built
    # once for all 2^k masks: the marginal contributions below are
    # differences between entries of this table, so nothing is predicted
    # twice.
    coalition_values = np.empty((1 << n_features, n_explain), dtype=float)
    tiled_background = np.tile(matrix_background, (n_explain, 1))
    for mask in range(1 << n_features):
        revealed = [j for j in range(n_features) if mask >> j & 1]
        block = tiled_background.copy()
        if revealed:
            block[:, revealed] = np.repeat(matrix_explain[:, revealed], n_background, axis=0)
        prediction = np.asarray(model.predict(block), dtype=float)
        coalition_values[mask] = prediction.reshape(n_explain, n_background).mean(axis=1)

    # w(|S|) = |S|! (k - |S| - 1)! / k!, the weight that makes the
    # average over reveal ORDERS come out right. Precomputed per size
    # because it depends only on |S|.
    weights = np.array(
        [
            factorial(size) * factorial(n_features - size - 1) / factorial(n_features)
            for size in range(n_features)
        ],
        dtype=float,
    )
    values = np.zeros((n_explain, n_features), dtype=float)
    for mask in range(1 << n_features):
        size = int(mask).bit_count()
        if size == n_features:
            # The full coalition has nothing left to reveal, and its
            # weight is not defined -- w(k) would need (-1)!.
            continue
        weight = weights[size]
        for j in range(n_features):
            if mask >> j & 1:
                continue
            values[:, j] += weight * (coalition_values[mask | (1 << j)] - coalition_values[mask])

    base_value = float(coalition_values[0][0])
    prediction = coalition_values[(1 << n_features) - 1]
    efficiency_error = float(
        np.max(np.abs(values.sum(axis=1) - (prediction - base_value)))
    )
    if efficiency_error > EFFICIENCY_TOLERANCE_KW:
        raise ValueError(
            f"attributions miss the prediction by {efficiency_error:.3e} kW, above "
            f"{EFFICIENCY_TOLERANCE_KW:.0e}. The Shapley efficiency axiom is exact "
            "arithmetic, so this is not a tolerance to widen -- a coalition weight "
            "or a mask in this function is wrong."
        )

    explanation = ShapleyExplanation(
        label=label,
        feature_names=names,
        values_kw=values,
        base_value_kw=base_value,
        prediction_kw=prediction,
        n_background=n_background,
        efficiency_error_kw=efficiency_error,
        off_manifold_fraction=_off_manifold_fraction(explain, background, pairs),
    )
    logger.info(
        "%s: %d hours explained against %d background rows, base value %.1f kW; "
        "mean |attribution| %s",
        label or "shapley",
        n_explain,
        n_background,
        base_value,
        {name: round(value, 2) for name, value in explanation.mean_abs_kw.items()},
    )
    for pair_name, fraction in explanation.off_manifold_fraction.items():
        if fraction > 0.0:
            logger.warning(
                "%s: %.1f%% of the coalition rows behind these attributions put "
                "(%s) outside anything the building produced. The attribution is "
                "an average over the model's answers on weather that did not occur.",
                label or "shapley",
                100.0 * fraction,
                pair_name,
            )
    return explanation


def explanation_comparison(
    explanation: ShapleyExplanation,
    permutation_importance: Mapping[str, float],
) -> pd.DataFrame:
    """Put attribution and held-out importance side by side, with ranks.

    The two columns answer different questions and the frame is built so
    that is unmissable. `mean |phi|` is computed from the model's own
    outputs and never sees the measured load, so it exists for any model
    including one fitted to noise. The permutation column is the
    held-out RMSE cost of scrambling the feature, i.e. it is scored
    against the truth on hours the model did not see, so it can be zero
    or negative -- and it is exactly zero for a model that learnt
    nothing, which is the disagreement this table is for.

    Args:
        explanation: From `shapley_values_kw`.
        permutation_importance: From `hybrid.permutation_importance_kw`,
            over the same features.

    Returns:
        One row per feature, ordered by attribution rank.

    Raises:
        ValueError: If the two do not cover the same features.
    """
    attribution = explanation.mean_abs_kw
    if set(attribution) != set(permutation_importance):
        raise ValueError(
            f"features differ: attribution has {sorted(attribution)}, permutation "
            f"has {sorted(permutation_importance)}"
        )
    permutation_rank = {
        name: rank
        for rank, name in enumerate(
            sorted(permutation_importance, key=lambda n: permutation_importance[n], reverse=True),
            start=1,
        )
    }
    return pd.DataFrame(
        {
            "mean |phi| kW": [attribution[name] for name in attribution],
            "attribution rank": list(range(1, len(attribution) + 1)),
            "permutation kW": [permutation_importance[name] for name in attribution],
            "permutation rank": [permutation_rank[name] for name in attribution],
        },
        index=pd.Index(list(attribution), name="feature"),
    )


def rank_agreement(comparison: pd.DataFrame) -> float:
    """Spearman correlation between the two rankings in the frame.

    Reported as one number because a table of six rows invites reading
    the top line and stopping. +1 means the two methods order the
    features identically; near 0 means the model's internal accounting
    and its measured dependence on unseen data have nothing to do with
    each other, which happens whenever the model has little to be
    dependent on.

    Args:
        comparison: From `explanation_comparison`.

    Returns:
        Spearman rho over the two rank columns.
    """
    from scipy.stats import spearmanr

    result = spearmanr(
        comparison["attribution rank"].to_numpy(dtype=float),
        comparison["permutation rank"].to_numpy(dtype=float),
    )
    return float(result.statistic)


def _demo() -> None:
    """The same ensemble, on a real residual and on a shuffled one.

    The shuffled arm is the control this lesson turns on. Its target is
    the identical residual with the hours randomly reordered, so there is
    nothing left to learn -- every relationship to weather has been
    destroyed -- and the model is fitted on it anyway, exactly as it
    would be by someone who never checked.

    The explained model is fitted on EVERY hour and explained on hours it
    was fitted on. That is not a shortcut taken here; it is the standard
    attribution workflow -- you explain the model you shipped, using the
    data you have -- and its in-sample-ness is the reason the attribution
    cannot see memorisation. The two columns beside it are held out: the
    permutation cost is measured on unseen hours, and the out-of-fold
    R-squared is L7.3's ML share on this target.

    What to watch: the attribution ranking is produced for BOTH arms and
    looks equally purposeful in both, because it is computed from the
    model rather than from the building. The held-out columns collapse on
    the shuffled arm. One of these can tell you the model is worthless
    and the other cannot.
    """
    from cooling_twin import set_seed
    from cooling_twin.analysis.hybrid import (
        build_features,
        default_model_factory,
        out_of_fold_correction,
        permutation_importance_kw,
    )
    from cooling_twin.calibration.crossval import expanding_window_folds

    rng = set_seed()
    index = pd.date_range("2016-01-01", periods=8760, freq="h")
    steps = np.arange(index.size, dtype=float)
    t_outdoor = (
        20.0
        + 12.0 * np.sin(2.0 * np.pi * (steps - 2000.0) / 8760.0)
        + 5.0 * np.sin(2.0 * np.pi * steps / 24.0)
    )
    humidity = 0.006 + 0.004 * np.sin(2.0 * np.pi * (steps - 2000.0) / 8760.0)
    # The two switched terms L7.1c and Q11 actually measured, injected so
    # the answer is known before anything is fitted -- same construction
    # as `hybrid._demo`.
    residual = (
        40.0 * np.maximum(t_outdoor - 28.0, 0.0)
        + 25.0 * np.maximum(humidity * 1000.0 - 8.0, 0.0)
        + rng.normal(0.0, 90.0, size=index.size)
    )

    features = build_features(index, t_outdoor, humidity)
    matrix = features.to_numpy(dtype=float)
    folds = expanding_window_folds(index.size, n_folds=5, spin_up_hours=0, embargo_hours=168)
    background = sample_rows(features, 120)
    explain = sample_rows(features, 150, seed=SEED + 1)

    for name, target in (
        ("true residual", residual),
        ("shuffled residual", rng.permutation(residual)),
    ):
        model = default_model_factory()()
        model.fit(matrix, target)
        explanation = shapley_values_kw(model, explain, background, label=name)
        comparison = explanation_comparison(
            explanation, permutation_importance_kw(features, target, folds)
        )
        correction, scored = out_of_fold_correction(features, target, folds)
        held_out = target[scored]
        ss_residual = float(((held_out - correction[scored]) ** 2).sum())
        ss_total = float(((held_out - held_out.mean()) ** 2).sum())
        print(f"\n=== {name} ===")
        print(comparison.round(2).to_string())
        print(
            f"base value {explanation.base_value_kw:+.1f} kW   "
            f"efficiency error {explanation.efficiency_error_kw:.2e} kW   "
            f"rank agreement {rank_agreement(comparison):+.2f}"
        )
        print(
            f"out-of-fold R2 of the correction: {1.0 - ss_residual / ss_total:+.3f}"
            "   (this is the number the attribution cannot produce)"
        )
        print(
            "fabricated coalition rows: "
            + ", ".join(
                f"{pair} {100.0 * fraction:.0f}%"
                for pair, fraction in explanation.off_manifold_fraction.items()
            )
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    _demo()

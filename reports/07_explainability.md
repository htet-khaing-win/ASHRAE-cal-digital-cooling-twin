# What a feature-importance plot can and cannot tell you — measured, both buildings

> M7's explainability finding. The SAME learner is explained two ways on
> the same buildings: exact Shapley attribution, and the variance
> decomposition of L7.3. A third arm — the identical learner fitted on
> the residual with its hours **shuffled** — is run as a control, because
> a diagnostic that cannot collapse on data with nothing in it is not a
> check.
>
> **Training year only (ADR-002).** Nothing here touches 2017. Physics
> parameters are read frozen from the 2016 calibration artifacts and the
> optimiser is not imported. `Hog_education_Cathleen` is excluded per
> ADR-015 — an attribution over a booster relearning its clip-at-zero
> base load would describe the defect as an explanation.
>
> Reproduce:
> `python scripts/compare_explanations.py`
> Library demo (synthetic, answer injected):
> `python -m cooling_twin.analysis.explain`
> Artifact: `reports/calibration_runs/explanations_2016.json`
> Figure: `reports/figures/l7_4_explanation_comparison.png`

## Summary

Three findings, in order of how much they change what may be claimed.

1. **Two importance methods agreeing is not evidence that the model is
   any good.** On `Bull_education_Luke` the attribution ranking and the
   held-out permutation ranking agree **perfectly** (Spearman ρ =
   **1.00**), both name outdoor dry bulb first, and both look
   substantial — 259.2 kW attributed, 67.0 kW of held-out RMSE cost. The
   same correction is worth **1.08%** of measured variance out of fold,
   against **6.84%** in sample, and it makes **3 of 5** folds worse.
   Agreement measures whether the two methods see the same model. It
   says nothing about whether the model saw the building.
2. **A model with nothing to learn still produces a confident plot.**
   The shuffled control retains **20%** of the real attribution
   magnitude on Claude and **15%** on Luke, and it produces a top-ranked
   feature in both cases — on Luke, `humidity_ratio_g_per_kg` at 39.3 kW,
   a finding about a target that was destroyed by construction. The
   held-out columns of the same control collapse to ≤ 2.4 kW, and its
   out-of-fold share is negative.
3. **On Claude the attribution does not name the term the physics
   diagnosis found.** L7.1c located Claude's defect as a hockey stick in
   *instantaneous* outdoor temperature: flat below ~16 °C, +54.4 kW/K
   above it. The attribution ranks `outdoor_dry_bulb_c` **fourth**
   (92.2 kW), behind its own 24-hour mean (330.9 kW), humidity
   (234.4 kW) and `hour_cos` (124.9 kW). The two temperature columns
   correlate, so the split between them is a property of the fit, not of
   the building. A reader who takes the top bar as the missing physics
   would go and size a lagged-weather term.

The one number in this report that answers "is the model any good" is
the ML share, and it is the only one produced by comparing the model to
the meter.

## What was fitted

Four models per building, on the 2016 training year:

| # | model | fitted on | used for |
|---|---|---|---|
| 1 | calibrated 2R2C physics | — (read frozen) | the residual everything else explains |
| 2 | out-of-fold correction | each fold's past | the ML share (L7.3) |
| 3 | **deployment** correction | **every hour** | the attribution |
| 4 | **shuffled control** | every hour, hours permuted | the control arm |

Model 3 is fitted on all hours and explained on hours it was fitted on.
That is not a shortcut: it is the standard attribution workflow — you
explain the model you shipped — and its in-sample character is precisely
why the attribution cannot see memorisation.

Attribution is **exact** interventional Shapley, all 2⁶ = 64 coalitions
enumerated (`cooling_twin.analysis.explain`), 150 explained hours
against 120 background hours, both drawn under `SEED`. No sampling
approximation is involved: the efficiency identity
`Σφ = prediction − base value` closes to 3.4e-13 kW on Claude and
3.7e-13 kW on Luke.

## Fox_education_Claude — 8,782 hours, 6,479 scored

Physics **90.6%** / ML **3.3%** out of fold / unexplained **6.1%**.
In-sample ML 6.3%, i.e. a memorisation gap of **+3.0 pp**.
Per-fold ML share: `+11.95  +2.87  +38.14  +0.12  −3.13` — a seasonal
correction carried by fold 3.

| feature | attribution, kW | held-out permutation, kW | control attribution, kW |
|---|---:|---:|---:|
| `outdoor_dry_bulb_24h_mean_c` | **330.9** | 25.45 | 58.6 |
| `humidity_ratio_g_per_kg` | 234.4 | **35.13** | 34.0 |
| `hour_cos` | 124.9 | 32.13 | 17.7 |
| `outdoor_dry_bulb_c` | 92.2 | 20.91 | 37.9 |
| `hour_sin` | 37.9 | 6.38 | 14.7 |
| `is_weekend` | 19.1 | 1.97 | 7.5 |
| **total** | **839.3** | — | **170.5 (20%)** |

Rank agreement ρ = **0.83**: the two methods disagree about which
feature matters most. Base value +68.3 kW (the correction's mean
prediction over the background); on the control, −1.7 kW.

## Bull_education_Luke — 8,572 hours, 6,304 scored

Physics **86.7%** / ML **1.1%** out of fold / unexplained **12.2%**.
In-sample ML 6.8%, a memorisation gap of **+5.8 pp** — the largest in
the project. Per-fold ML share: `+5.94  −1.07  −19.64  +5.47  −0.99`.

| feature | attribution, kW | held-out permutation, kW | control attribution, kW |
|---|---:|---:|---:|
| `outdoor_dry_bulb_c` | **259.2** | **66.98** | 25.9 |
| `outdoor_dry_bulb_24h_mean_c` | 242.1 | 66.37 | 25.9 |
| `humidity_ratio_g_per_kg` | 133.8 | 49.10 | **39.3** |
| `hour_sin` | 67.3 | 13.49 | 8.4 |
| `hour_cos` | 32.9 | 10.42 | 10.1 |
| `is_weekend` | 25.8 | 4.85 | 3.9 |
| **total** | **761.2** | — | **113.5 (15%)** |

Rank agreement ρ = **1.00**. This is the report's central exhibit: two
importance methods, one of them held out, in perfect agreement about a
correction worth 1.1% out of fold that harms three folds in five.

Why the held-out permutation column is large here and the ML share is
not: permutation importance is measured **relative to the fitted model's
own error**. It answers "does this model depend on this feature", which
a model can do strongly while still failing to beat no correction at
all. Only the variance decomposition compares the prediction to the
meter.

## The fabricated-input cost

Interventional Shapley pastes the explained hour's value for a coalition
onto background rows and leaves the rest. Where two features are
physically coupled, that produces inputs the building never produced:

| building | `outdoor_dry_bulb_c` vs its 24-hour mean | `hour_sin` vs `hour_cos` |
|---|---:|---:|
| Fox_education_Claude | **26.0%** | 10.2% |
| Bull_education_Luke | 11.4% | 10.0% |

A quarter of the coalition rows behind Claude's attribution ask the
model about a temperature that did not belong to its own week. The
model answers, because a tree ensemble always answers. This is inherent
to the interventional form — which is also what makes a feature the
model ignores score exactly zero — and it is reported rather than
resolved.

## What may be claimed

- ✅ "Physics explains 90.6% of the measured variance on the primary
  building; the learnt correction adds 3.3% out of fold, concentrated in
  summer; 6.1% is unexplained." Checkable, and produced against the
  meter.
- ✅ "Attribution on the deployed correction ranks the 24-hour mean
  temperature first, but the held-out cost ranks humidity first, and the
  correction's total value is 3.3%."
- ❌ "SHAP shows the model has learnt that lagged temperature drives the
  residual." The control model, which learnt nothing, ranks the same
  feature first on Claude.
- ❌ "Both importance methods agree, so the correction is sound." Luke,
  ρ = 1.00, 1.1% out of fold.

## Limitations

1. The attribution is computed on **150 hours** against a **120-hour**
   background, and it is a sample estimate of a whole-year quantity.
   Measured, on Claude, by raising the counts to 400/240: the **ordering
   is unchanged**, but the magnitudes move by up to 11%
   (`humidity_ratio_g_per_kg` 234.4 → 209.5 kW, total 839.3 → 798.4 kW)
   and the **base value moves by half** (+68.3 → +34.5 kW), because the
   background sample is what defines it. Quote an attribution with its
   counts, and never to three significant figures.
2. The control arm destroys the target's time structure completely.
   That is the right null for "did this model learn anything about the
   building", and the wrong null for "did it learn something a simpler
   model could not". The latter is the L6.4 baseline comparison.
3. Attribution magnitudes are relative to the base value, which is
   itself a property of the background sample. Two teams with different
   backgrounds will publish different kilowatts for the same model.
4. Everything here is the training year. The ML shares transfer
   differently across years — see the L7.3a re-read log in
   `07_PROGRESS.md`, where the correction was worth more across years
   than within one, for a reason (expanding-window bias) that is stated
   there.

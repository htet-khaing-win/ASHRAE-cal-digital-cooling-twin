# Calibration — training year (2016)

> ⚠️ **This is not the M6 gate report.** Every number here is from the
> **training** year. The 2017 test set has not been opened; L6.10 opens it once,
> deliberately, and logs the access in `07_PROGRESS.md`. A model that passes on
> its training year has not passed the gate.

*Generated 2026-08-13. Model: 3-node inverse RC with the ADR-011 ventilation
term. 5 calibrated parameters. Objective: L6.6's G14 budget with the clipping
penalty. Seed 42, differential evolution then L-BFGS-B.*

## Headline

All three selected buildings meet both G14 hourly criteria on the training
year, and all three beat their linear-regression baseline. This is the first
point in the project where the physics model has beaten a baseline at all.

## Before / after the ADR-012 bound amendment

"Before" used a hand-set `internal_gain_w_per_m2` upper bound anchored on the
building's electricity meter. "After" derives that bound from the data — 4× the
cooling load that survives the coldest hours (`calibration/bounds.py`). Nothing
else changed: same model, same objective, same seed, same optimiser settings.

| Building | Bound before | CV(RMSE) before | G14 before | Bound after (derived) | CV(RMSE) after | G14 after |
|---|---|---|---|---|---|---|
| `Fox_education_Claude` | 120.0 (hand-set) | 52.46% | ❌ FAIL | **711.4** | **11.72%** | ✅ PASS |
| `Bull_education_Luke` | 120.0 (hand-set) | 16.19% | ✅ PASS | **276.8** | **14.13%** | ✅ PASS |
| `Hog_education_Cathleen` | 120.0 (hand-set) | 28.65% | ✅ PASS | **127.3** | **28.66%** | ✅ PASS |

NMBE is 0.00% in every "after" run.

The amendment mattered where the hand-set bound was binding (Claude, where the
fit was starved by a factor of 2.6 and under-predicted by 50.7%), helped a
little where it was marginal (Luke), and changed nothing where it was never
binding (Cathleen). That is the behaviour a correct bound should have.

## Against the baselines

| Building | Annual mean | Linear regression | **Calibrated RC** | G14 | Relative improvement on best baseline |
|---|---|---|---|---|---|
| `Fox_education_Claude` | 40.75% | 14.76% | **11.72%** | ✅ | 20.6% |
| `Bull_education_Luke` | 40.11% | 20.88% | **14.13%** | ✅ | 32.3% |
| `Hog_education_Cathleen` | 69.09% | 38.85% | **28.66%** | ✅ | 26.2% |

⚠️ The M6 gate's supporting requirement is a **≥ 30% relative** improvement on
the best baseline. Only `Bull_education_Luke` meets it. Passing G14 while
barely out-performing a two-parameter straight line is a weak result and must
be reported as one.

## Calibrated parameters

| Parameter | Claude | Luke | Cathleen | Bound |
|---|---|---|---|---|
| `ua_envelope_w_per_m2k` | **3.00** ⚠ | 1.22 | 0.34 | 0.3–3.0 |
| `r_internal_ratio` | 18.57 | 3.65 | 11.65 | 2–20 |
| `internal_gain_w_per_m2` | 372.0 | 151.5 | 109.2 | 1–auto |
| `vent_flow_kg_per_s` | **179.6** ⚠ | 56.4 | 40.9 | 0.5–180 |
| `t_setpoint_c` | 24.83 | 24.26 | 21.78 | 20–26 |

⚠ = pinned at a bound. Claude pins **two** parameters (envelope UA at its
ceiling, ventilation flow at 10 ACH), which says the fit still wants more
weather-driven load than the physical bounds allow even though it now passes.
Read alongside Q8: this building runs simultaneous heating and cooling for
99.5% of the year, so its cooling load is not a straightforward function of
outdoor conditions.

### Interpretability caveat

`internal_gain_w_per_m2` at 372 W/m² for Claude cannot be read as "internal
gains". The building's own electricity is 53.6 W/m². The parameter is absorbing
a constant coil load, and Q8's reheat evidence suggests what produces it.
Naming it "internal gain" is now a known misnomer for reheat-dominated
buildings — carried into M7 rather than papered over.

### Equifinality — how much of that table is actually measured

The parameters above are point estimates from one optimisation. L6.8 probed
`Fox_education_Claude` with 41 independent local refinements to find every
parameter set the data **cannot reject** — objective within 5% of the best,
which on this building means CV(RMSE) 11.68% to 12.16%, all passing G14.

Two studies, because the sampler decides which question gets answered:

| Study | Restarts drawn from | Behavioural sets | Question |
|---|---|---|---|
| whole box | 100% of each bound width | 3 of 41 | is there a **distant** rival that fits as well? |
| ridge | 10% of each bound width | 7 of 41 | how **wide** is the family around the reported answer? |

Whole-box answer: **no distant rival.** 38 of 41 refinements landed in other
basins entirely, objectives 0.414 to 11.85 against a 0.409 threshold. The
behavioural region is small and isolated — which is also why that study could
not answer the second question, and why both were run.

Ridge study, `internal_gain_w_per_m2` ↔ `t_setpoint_c` at **r = +1.000**:

| Parameter | Behavioural range | % of bounds | Verdict |
|---|---|---|---|
| `ua_envelope_w_per_m2k` | 2.733 – 3.000 | 9.9% | identified |
| `r_internal_ratio` | 17.15 – 19.77 | 14.6% | weakly identified |
| `internal_gain_w_per_m2` | 348.0 – 387.1 | 5.5% | identified |
| `vent_flow_kg_per_s` | 174.6 – 179.6 | 2.8% | identified |
| `t_setpoint_c` | **22.90 – 26.00** | **51.6%** | **unidentified** |

The setpoint is not measured by this calibration at all: any value from 22.9 °C
to the top of its range fits equally well, provided internal gain moves with it.
The mechanism is physical, not numerical — a higher assumed setpoint means less
cooling driven by the indoor-outdoor difference, so the fit buys the shortfall
back as internal gain. Reporting `t_setpoint_c = 24.83` without that range is a
precision the data does not support.

In the whole-box study, run against the same calibration, both pinned
parameters (`ua_envelope`, `vent_flow`) sat on their ceiling in **every**
behavioural set. That is reported as `bound-limited`, not as `identified`: a
span of zero against a bound means the box stopped the fit, not that the data
determined the value.

**Consequence for advice.** The same retrofit priced by every admissible
parameter set:

| Measure | Calibrated answer | Across behavioural sets |
|---|---|---|
| ventilation setback −30% | 0.86% saving | −0.06% to 2.31% |
| envelope upgrade −30% UA | 0.03% saving | −0.23% to 0.46% |

Both measures are within noise of doing nothing under *some* admissible
parameter set. No retrofit recommendation can be made for this building from
this calibration, and that conclusion is not visible anywhere in the CV(RMSE).

```bash
python scripts/run_equifinality.py --n-starts 40 --start-spread 1.00
python scripts/run_equifinality.py --n-starts 40 --start-spread 0.10
```

Artifacts: `reports/calibration_runs/equifinality_*_spread{100,010}.json`,
figures `reports/figures/l6_8_equifinality{,_wholebox}.png`. Each study records
**every** refinement, so it can be re-thresholded or redrawn without re-running
(`--replot`). Not yet run for Luke or Cathleen — an M6 gap, listed as such.

## Reproducing

```bash
python scripts/run_calibration.py --config config/calibration.yaml
```

Each run writes a JSON artifact to `reports/calibration_runs/`, carrying the
seed, both stage objectives, the derived bounds, evaluation counts and the
metric decomposition. Ten runs are on record, including the superseded ones.

# Calibration and the M6 gate — Fox, Bull, Hog

> 🔓 **The test set is open.** 2017 was read once, on 2026-08-13, by
> `scripts/open_test_set.py --open-test-set`. The access is logged in
> `07_PROGRESS.md`. Nothing was refitted on 2017: the script never imports the
> optimiser, and the parameters are read frozen from the 2016 artifacts.

*Model: 3-node inverse RC with the ADR-011 ventilation term. 5 calibrated
parameters. Objective: L6.6's G14 budget with the clipping penalty. Seed 42,
differential evolution then L-BFGS-B, fitted on 2016 only.*

## THE GATE — held-out year, 2017

**Result: the primary gate FAILS. Two of three buildings pass; the gate
requires all three.** The 2-of-3 outcome was accepted on 2026-08-13
(ADR-014) and `Hog_education_Cathleen` is retired to `negative_case` in
`config/buildings.yaml`. Accepting the outcome does **not** tick the
gate: the criterion in `06_ASSESSMENT.md` is unchanged and unmet.

| Building | Train NMBE | Train CV(RMSE) | Test NMBE | Test CV(RMSE) | G14 |
|---|---|---|---|---|---|
| `Fox_education_Claude` (primary) | −0.00% | 11.72% | **+3.10%** | **11.58%** | ✅ PASS |
| `Bull_education_Luke` | −0.00% | 14.13% | **+1.35%** | **11.14%** | ✅ PASS |
| `Hog_education_Cathleen` | −0.00% | 28.66% | **−1.58%** | **31.65%** | ❌ FAIL |

Test-year CV(RMSE) against the baselines, all three fitted on **2016** and
evaluated on 2017 through their stored coefficients:

| Building | Annual mean | Linear regression | **Calibrated RC** | Relative improvement |
|---|---|---|---|---|
| `Fox_education_Claude` | 41.00% | 15.71% | **11.58%** | 26.3% |
| `Bull_education_Luke` | 35.77% | 18.59% | **11.14%** | **40.1%** |
| `Hog_education_Cathleen` | 68.54% | 40.70% | **31.65%** | 22.2% |

### What passed, and by how much

Two buildings did **better** on the held-out year than on the year they were
fitted to — Claude 11.72% → 11.58%, Luke 14.13% → 11.14%. That is not a
paradox and it is not luck: it is what a model with no overfitting looks like
when the test year happens to be slightly easier. It is also the strongest
available evidence that the calibration captured something about the buildings
rather than about 2016's weather.

Luke is the one building meeting every requirement: G14 on both years, and a
40.1% relative improvement on its best baseline, clearing the ≥30% supporting
requirement.

### Why Cathleen failed — structure, not overfitting

The mechanical reading of `06_ASSESSMENT.md`'s signature table is "good on
train, poor on test → overfitting → remove capacity". That reading is **wrong
here**, and the evidence is the seasonal decomposition on both years:

| Season | Cathleen 2016 NMBE (train) | Cathleen 2017 NMBE (test) |
|---|---|---|
| winter (DJF) | **+25.4%** | **+19.8%** |
| spring (MAM) | **−23.2%** | **−27.8%** |
| summer (JJA) | +10.9% | +9.4% |
| autumn (SON) | −14.6% | −13.5% |
| **annual** | **−0.00%** | **−1.53%** |

The fault is the same size on the training year. Overfitting is a fault that
appears on data the fit never saw; this one was always there, hidden by an
annual NMBE that averages a +25% winter against a −23% spring. The prescribed
actions are opposite — overfitting says *remove* model capacity, structure
error says *add* it — so the distinction decides what happens next.

Second contributing fact: Cathleen's training fit left only **1.34 points of
headroom** under the 30% limit. It was never comfortably inside the standard,
and the ordinary difference between two weather years (+2.99 pp) put it over.
Reporting 28.66% as a pass without that caveat was optimistic, and L6.9's
cross-validation would have said so had it been run on this building.

### The same signature on the primary building

Claude's test-year seasonal NMBE runs winter −7.59%, spring −2.78%, **summer
+10.70%**, autumn +2.87%. That is the fold-2 finding
(`reports/05_fold2_diagnosis.md`) reproducing on data the model has never seen:
the load-versus-temperature response is too flat, so summer is under-predicted.
It sits just inside G14 annually and just outside the ±10% NMBE limit in summer
alone. Predicted before the test set was opened, and confirmed by it.

### Supporting requirements

| Requirement | Status |
|---|---|
| Naive baselines computed | ✅ annual mean + linear regression, fitted 2016 |
| Beats best baseline by ≥30% relative CV(RMSE) | ❌ **Luke only** (40.1%; Claude 26.3%, Cathleen 22.2%) |
| Morris/Sobol sensitivity performed | ✅ L6.5, Morris, 7 screened → 4 kept (5 with ADR-011) |
| ≤10 parameters calibrated | ✅ 5 |
| All parameters within physical bounds | ⚠️ within, but Claude and Cathleen sit ON bounds (see below) |
| `metrics.py` 100% coverage, known-answer tests | ✅ |
| Seed fixed, reproducible | ✅ seed 42; the L6.9 re-run reproduced exactly |
| Run artifact JSON saved | ✅ `reports/calibration_runs/gate_2017_opened.json` |
| `reports/02_calibration.md` written | ✅ this file |

### The decision, and what it costs — ADR-014

Three options existed. Only two were available.

| Option | Verdict |
|---|---|
| (a) Accept 2-of-3, document Cathleen | **Chosen.** Costs the headline, keeps the evidence intact. |
| (b) Fix the structure error, re-evaluate | Legitimate — and it is M7's first task anyway. But any 2017 number produced *after* a model change is a re-read, not a clean held-out result, and must be labelled as one. |
| (c) Re-select a third building | **Refused.** With 2017 open, choosing a replacement means choosing the building the test set has already approved. |

Cathleen is kept rather than deleted because it is a negative case reached by
the **full** process — selected, cleaned, screened, calibrated,
cross-validated, gated — where `Fox_education_Theodore` was screened out before
any modelling began. A failure that survives the whole pipeline says more about
the method's limits than one caught by a filter.

### A note on the selection screen

ADR-012's screen gave Cathleen a weather-explainability ceiling of 17.4%
against a 30% limit — 42% headroom — and it failed anyway at 31.65%. The screen
was not wrong. **A ceiling is a lower bound on achievable error, not a
prediction of success:** it says a perfect weather-driven model could reach
17.4%, and the fitted model landed 14 points above its own floor. Clearing the
screen is necessary and not sufficient. Recorded in `config/buildings.yaml`
under `screen_said`, because that misreading is the one that just cost a
building.

### Honest summary

A digital twin of `Fox_education_Claude` and `Bull_education_Luke` meets ASHRAE
Guideline 14's hourly criteria on a held-out year, beating a two-parameter
linear regression by 26% and 40% respectively — with both buildings scoring
*better* on the year they had never seen than on the year they were fitted to.
A third building of the same type, at a different site, does not, and the
reason is a seasonal structural error that the annual metrics on its training
year concealed.

**The claim, in the only form the evidence supports:** the methodology
transferred to two of three buildings attempted, with a measured and explained
failure on the third. Not "the method transfers to buildings of this type" —
that sentence is not available, and the third building is why.

---

# Training-year detail (2016)

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

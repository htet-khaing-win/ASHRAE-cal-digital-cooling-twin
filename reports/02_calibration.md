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

## Reproducing

```bash
python scripts/run_calibration.py --config config/calibration.yaml
```

Each run writes a JSON artifact to `reports/calibration_runs/`, carrying the
seed, both stage objectives, the derived bounds, evaluation counts and the
metric decomposition. Ten runs are on record, including the superseded ones.

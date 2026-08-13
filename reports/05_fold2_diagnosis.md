# Why cross-validation fold 2 fails — Fox_education_Claude, 2016

> L6.9b's 4-fold within-2016 cross-validation passed on three folds and
> failed on one. This is the diagnosis of that fold. **2016 only** — the
> held-out year is untouched (ADR-002).
>
> Reproduce: `python scripts/diagnose_crossval_fold.py --fold 2`
> Artifact: `reports/calibration_runs/fold2_diagnosis_Fox_education_Claude_2016.json`
> Figure: `reports/figures/l6_9_fold_residuals.png`

## The failure

| Fold | Validation window | Mean load | CV(RMSE) | NMBE | G14 |
|---|---|---|---|---|---|
| 1 | 14 Mar – 26 May | 6,024 kW | 11.78% | +5.77% | PASS |
| **2** | **26 May – 07 Aug** | **10,176 kW** | **15.68%** | **+11.93%** | **FAIL (NMBE)** |
| 3 | 07 Aug – 19 Oct | 8,867 kW | 9.30% | +6.07% | PASS |
| 4 | 19 Oct – 31 Dec | 5,104 kW | 8.01% | −3.35% | PASS |

Fold 2 trains on 1 Jan – 26 May and is asked about peak Tempe summer.
Positive NMBE means the model **under-predicts**: measured cooling
exceeds predicted by ~12%, about 1,210 kW on average.

## Verdict

**The bias is a constant-term error caused by extrapolation, not by the
pinned weather parameters and not by reheat.** A second, separate fault
— a systematically too-flat weather response — is real, is present in
every fold, and is *not* what makes fold 2 fail.

### Evidence 1 — the same window, a different parameter set

| Parameters used on fold 2's window | CV(RMSE) | NMBE |
|---|---|---|
| fold 2's own (trained Jan–May) | 15.68% | +11.93% ❌ |
| full-year (trained Jan–Dec) | 12.31% | +6.91% ✅ |

The model structure *can* represent a Tempe summer inside G14. The fold
does not fail because the form is incapable — it fails because of what
its training window contained.

### Evidence 2 — one parameter at a time

Each row swaps a single fold-2 parameter to its full-year value and
re-scores the same window:

| Swapped | From | To | CV(RMSE) | NMBE | Bias removed |
|---|---|---|---|---|---|
| **`internal_gain_w_per_m2`** | 293.84 | 372.05 | **10.38%** | **−1.91%** | **10.01 pp** |
| `ua_envelope_w_per_m2k` | 2.95 | 3.00 | 15.59% | +11.81% | 0.11 pp |
| `r_internal_ratio` | 17.56 | 18.57 | 15.67% | +11.91% | 0.01 pp |
| `vent_flow_kg_per_s` | 179.93 | 179.60 | 15.73% | +11.98% | −0.05 pp |
| `t_setpoint_c` | 20.93 | 24.83 | 23.15% | +20.81% | −8.88 pp |

One parameter carries the entire failure. Correcting the constant term
alone turns a G14 failure into a comfortable pass **and** improves
CV(RMSE) by 5.3 points. The two pinned parameters account for 0.11 pp
and −0.05 pp — nothing.

The last row is the important corroboration: moving `t_setpoint_c`
*alone* makes the fit dramatically **worse**. That is the L6.8
compensating pair (`internal_gain` ↔ `t_setpoint`, r = +1.000) behaving
exactly as measured — the two are only meaningful together, and fold 2
sits at the low-gain / low-setpoint end of that ridge:

| Fold | `internal_gain` | `t_setpoint` |
|---|---|---|
| 1 | 305.96 | 23.04 |
| **2** | **293.84** | **20.93** |
| 3 | 349.56 | 23.52 |
| 4 | 379.94 | 25.29 |
| full year | 372.05 | 24.83 |

Fold 2's training window (January to May) contains no hours that pin the
constant term at its annual value, so the fit slides down the ridge and
locks in a constant ~78 W/m² too small. Every hour of the summer
validation window then inherits that deficit.

### Evidence 3 — reheat is not the cause

`corr(fold-2 residual, hot-water meter) = +0.112`. Negligible. Q8's
reheat hypothesis concerns the building's constant cooling floor and may
well be what the `internal_gain` parameter is physically absorbing — but
it does **not** explain this fold's summer bias.

## The second, separate fault

Removing the bias does not remove the residual. Decomposing fold 2's
residual before and after the winning swap:

| | Mean residual | Intercept | Slope vs temp | Slope vs humidity |
|---|---|---|---|---|
| fold-2 parameters | +1,210 kW | −1,479 kW | **+51.50 kW/K** | **+113.25 kW per g/kg** |
| after `internal_gain` swap | −194 kW | −2,884 kW | **+51.50 kW/K** | **+113.25 kW per g/kg** |

The slopes are *identical* — arithmetically inevitable, since a constant
only moves the intercept, and that is precisely the point: **the bias
and the shape are two different faults.** The constant term fixed the
first and left the second untouched.

That second fault is visible in every fold (see the figure: all four
binned-mean lines slope upward with outdoor temperature) and in the
full-year fit scored season by season:

| Season | Hours | Mean load | CV(RMSE) | NMBE |
|---|---|---|---|---|
| winter (DJF) | 2,183 | 4,059 kW | 14.57% | **−9.81%** (over-predicts) |
| spring (MAM) | 2,208 | 5,938 kW | 11.77% | −8.33% |
| summer (JJA) | 2,207 | 10,326 kW | 11.86% | **+7.49%** (under-predicts) |
| autumn (SON) | 2,184 | 7,181 kW | 7.12% | +1.62% |

Over-predicting the cool months and under-predicting the hot ones is one
signature, not two: **the model's load-versus-temperature response is
too flat.** The annual fit hides it by splitting the difference — each
season individually sits inside G14, and the year as a whole scores
NMBE −0.00%.

This is where the pinned parameters do matter. The unexplained slope is
~51.5 kW/K against a modelled steady-state sensible slope of roughly
232 kW/K (ventilation 179.9 kg/s × 1.006 kJ/kg·K ≈ 181 kW/K, plus the
envelope path `UA·A·ratio/(ratio+1)` ≈ 51 kW/K) — about 22% more slope
than the model produces. Supplying it from the terms available would
need ventilation at ~231 kg/s (12.9 ACH, against a 10 ACH bound) or
roughly double the envelope UA. **Both are outside their physical
bounds, and both already sit on their ceilings** in the full-year fit.
So the too-flat response cannot be fixed by re-fitting; it needs a term
the model does not have.

## What this means for L6.10

1. **Fold 2's failure does not indicate overfitting.** The mean gap
   across folds is +0.33 pp; the failure is a training-window coverage
   problem specific to an expanding-window scheme whose earliest folds
   cannot contain a summer. The full-year calibration that L6.10 will
   evaluate is trained on all of 2016 and does not have this deficit.
2. **It does say something about deployment.** A model calibrated on a
   partial year, or on a year whose extremes differ from the year it is
   applied to, will carry a constant-term error of this size. That is a
   transferability statement, and it belongs in the report.
3. **The too-flat weather response is a genuine open limitation** and is
   the same structural finding as Q7/Q8, reached from a third direction.
   It should be stated in `02_calibration.md` before the gate, not
   discovered after it.
4. **No parameter, bound or model change is warranted on this evidence
   before L6.10.** Widening the ventilation bound to chase 51.5 kW/K
   would be fitting the bound to the residual — the exact move ADR-013
   exists to prevent.

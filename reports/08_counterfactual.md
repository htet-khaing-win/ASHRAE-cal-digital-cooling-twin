# What if — counterfactuals, and what they are worth

> M8's closing report. Three things are established here, in this order:
> that the twin answers a question regression cannot (L8.1), what it says
> when asked (L8.2), and how wide the answer's error bars are (L8.3).
> L8.4 is the last section: the same results written in the only language
> the evidence supports.
>
> **Training year only (ADR-002).** A counterfactual has no ground truth
> on any year — the world in which the setpoint was 1 K higher was never
> recorded — so opening 2017 for it would spend the project's scarcest
> asset on a question no data can settle.
>
> **THE ONE-LINE CAVEAT, and it applies to every energy number below.**
> The cooling-**load** model is calibrated and was tested on the held-out
> year (M6). The conversion from cooling load to **electricity** is not
> calibrated, and cannot be from this dataset: BDG2 records one thermal
> meter and one whole-building electricity meter, and no chiller
> sub-meter. Every kWh here is the twin's calibrated load passed through
> a **documented generic plant** (`config/plant.yaml`), never through the
> plant that is actually installed.
>
> Reproduce:
> `python scripts/compare_correlation_intervention.py`,
> `python scripts/run_counterfactual.py`,
> `python scripts/validate_intervals.py`
> Artifacts: `reports/calibration_runs/correlation_vs_intervention_2016.json`,
> `counterfactual_2016.json`, `conformal_coverage_2016.json`
> Figures: `l8_1_correlation_vs_intervention.png`, `l8_2_scenarios.png`,
> `l8_2_chiller_pump_tradeoff.png`, `l8_3_conformal_coverage.png`
> Notebook: `notebooks/06_counterfactual.ipynb` (L8.1 narrative)
>
> Companion reports: `02_calibration.md` (what was validated),
> `03_residual_analysis.md` (what the model still gets wrong).

## Summary

1. **Correlation and intervention disagree by a factor of six.** On
   Fox_education_Claude the observational slope of load on outdoor
   humidity is **507.6 kW per g/kg**; adjusting for outdoor dry bulb
   drops it to **239.2**; the twin's `do(w + 1 g/kg)` answer is
   **83.0**. On Bull_education_Luke: **187.1 → 129.0 → 94.5**. In a
   synthetic world where the causal truth is known by construction, the
   naive slope overstates it by **5.8×** and the closed-form
   contamination term predicts the gap to within 2%.
2. **The chiller and the pump can be made to disagree by one
   assumption.** Raising chilled-water supply by 2 K changes Claude's
   annual plant electricity by **−4.21%** if the coils are rebalanced so
   delta-T holds, and by **+5.10%** if the return temperature stays
   where it is. Same intervention, same plant, same year; the sign flips
   on whether one number moves. The pump's penalty is exactly
   `1.5³ − 1 = +237.5%`, which is the affinity law and not a fitted
   result.
3. **Load savings do not convert one-for-one into electricity savings.**
   `do(setpoint + 1 K)` cuts Luke's cooling **load** by 2.71% and its
   chiller **electricity** by only 1.48%, because a smaller load runs
   the plant further down its part-load curve. Anyone quoting a load
   reduction as an energy saving is over-claiming by ~45% on this
   building.
4. **The dominant uncertainty is not statistical.** Claude's ventilation
   setback scenario reads **−1.41%** at the calibrated parameters, and
   **−5.57% to −0.72%** across L6.8's three behavioural parameter sets —
   sets that fit the meter equally well. The week-block bootstrap
   interval on the same scenario, **[−3.55%, +0.29%]**, already contains
   zero. The scenario is not distinguishable from doing nothing.
5. **Conformal coverage reaches the 90% target, but only under a split
   whose assumption the data supports.** Under interleaved week blocks:
   Claude **93.5%**, Luke **90.1%**. Under a deployment-style forward
   split Luke reads **86.3%**, and running the same split backwards on
   Claude collapses it to **49.9%**. The exchangeability assumption is
   the binding constraint, not the method.

## 1. The gap between correlation and intervention (L8.1)

A known-truth check first, because real data offers none. The synthetic
world is a DAG: temperature causes humidity, and both cause load. The
univariate slope of load on humidity is then
`b + a·c·var(T)/var(w)`, a closed form:

| quantity | value |
|---|---|
| true humidity effect | 120.0 kW per g/kg |
| naive regression slope | **699.9** |
| predicted contamination (closed form) | 580.9 |
| adjusted for temperature | 118.7 |
| overstatement | **5.83×** |

120.0 + 580.9 = 700.9 against a measured 699.9 — the bias is not an
illustration, it is arithmetic.

Adjustment recovers the truth here because the confounder is measured,
unique and linear. On a real building none of those three hold, which is
why the same three numbers on the real record are not a repeat of the
demonstration but a different situation:

| | Claude | Luke |
|---|---|---|
| observational, kW per g/kg | 507.6 | 187.1 |
| adjusted for outdoor dry bulb | 239.2 | 129.0 |
| **interventional, `do(w + 1 g/kg)`** | **83.0** | **94.5** |
| corr(T, w) in the record | 0.38 | 0.79 |
| mean measured load, kW | 6,883 | 2,642 |

Nothing here proves the interventional number is *correct* — the twin
can be wrong, and M7 documented several ways in which it is. What it
establishes is that the first two columns answer a different question,
and that reporting either as "the effect of humidity" reports the
weather along with it.

**And the setpoint has no observational answer at all.** BDG2 records no
setpoint; the calibrated `t_setpoint_c` is one constant for the year, so
its regression coefficient does not exist — `ols_slope()` raises rather
than returning a number. The twin still answers: **−231.6 kW per K** on
Claude, **−71.5 kW per K** on Luke. That is the argument for M8 in one
line.

## 2. Five interventions (L8.2)

The plant each load is served by, sized from the measured peak
(`config/plant.yaml`):

| | Claude | Luke |
|---|---|---|
| chillers | 4 × 4,978 kW | 4 × 3,438 kW |
| CHW pump, rated | 603 kW at 0.79 m³/s | 417 kW at 0.55 m³/s |
| baseline chiller / pump | 7,137 / 291 MWh | 3,228 / 36 MWh |
| baseline plant COP | 8.14 | 6.94 |

The pump draws 3.9% of Claude's plant energy against the 10% design
share it was sized to, because a variable-speed pump spends the year at
part flow and pays the cube of it. That is the affinity law doing what
L1.4 said it would, and it is why the pump only becomes interesting when
something forces the flow up.

### Fox_education_Claude

| scenario | load % | chiller % | pump % | **plant %** | ±MWh/yr | bootstrap 90% | equifinal range |
|---|---|---|---|---|---|---|---|
| zone setpoint +1 K | −3.37 | −2.91 | −8.18 | **−3.12** | −232 | [−3.37, −2.90] | [−3.15, −3.10] |
| zone setpoint −1 K | +3.36 | +3.00 | +8.68 | **+3.23** | +240 | [+3.02, +3.45] | [+3.21, +3.23] |
| CHW +2 K, coils hold ΔT | 0.00 | −4.38 | 0.00 | **−4.21** | −313 | [−4.76, −3.79] | [−4.22, −4.21] |
| CHW +2 K, return fixed | 0.00 | −4.38 | **+237.50** | **+5.10** | +379 | [+4.07, +6.43] | [+5.09, +5.10] |
| ventilation setback −30% | −0.86 | −0.95 | −12.56 | **−1.41** | −105 | [−3.55, **+0.29**] | [**−5.57**, −0.72] |

### Bull_education_Luke

| scenario | load % | chiller % | pump % | **plant %** | ±MWh/yr | bootstrap 90% |
|---|---|---|---|---|---|---|
| zone setpoint +1 K | −2.71 | −1.48 | −6.41 | **−1.53** | −50 | [−1.78, −1.31] |
| zone setpoint −1 K | +2.71 | +1.35 | +6.74 | **+1.41** | +46 | [+1.20, +1.63] |
| CHW +2 K, coils hold ΔT | 0.00 | −3.39 | 0.00 | **−3.35** | −109 | [−3.72, −3.04] |
| CHW +2 K, return fixed | 0.00 | −3.39 | +237.50 | **−0.76** | −25 | [−1.10, −0.37] |
| ventilation setback −30% | −3.90 | −6.81 | −21.98 | **−6.98** | −228 | [−8.94, −5.37] |

Luke has no equifinality study, so its structural interval is
**unavailable — which is not the same as small**, and is the reason its
column is absent rather than blank.

Three things in these tables are worth more than the numbers themselves:

**The load column is not the energy column.** Luke's +1 K gives −2.71%
load and −1.48% chiller. The load reduction moves the plant to a lower
part-load ratio where L5.2's EIRFPLR curve is worse, and roughly 45% of
the thermal saving is eaten on the way to the meter. Claude loses less
(−3.37 → −2.91) because its plant sits closer to the efficient band to
begin with.

**The same intervention has opposite signs on the two buildings.**
`CHW +2 K, return fixed` costs Claude 5.10% and saves Luke 0.76%. The
mechanism is entirely the pump's share: Claude's pump is 3.9% of plant
energy and Luke's is 1.1%, so tripling it hurts Claude three and a half
times as much. Nothing about the buildings' physics differs here — the
difference is which component the answer is sensitive to.

**One scenario is indistinguishable from doing nothing.** Claude's
ventilation setback reads −1.41% with a bootstrap interval containing
zero and an equifinal range four times as wide as the point estimate. It
is reported because it was run, not because it is a recommendation.

### The chiller-versus-pump trade-off, swept

`reports/figures/l8_2_chiller_pump_tradeoff.png` sweeps the chilled-water
supply setpoint from 0 to +3.3 K — stopping there because
`03_DOMAIN_REFERENCE.md` §1 gives 6.7–10.0 °C as this setpoint's
documented optimisation range, and a plant model asked past its own
documented range is fiction.

With the coils rebalanced, the curve never turns: the chiller keeps
gaining and the total reaches **−6.5%** at +3.3 K on Claude and −5.5% on
Luke. With the return temperature fixed, the pump catches the chiller
and passes it:

| break-even, CHW supply increase | pump = 8% of HVAC | 10% (config) | 12% |
|---|---|---|---|
| Claude | +0.70 K | below +0.5 K | below +0.5 K |
| Luke | +2.69 K | +2.39 K | +2.09 K |

The pump share is a single documented band (§1: CHW pump 8–12% of HVAC
energy) and it moves Claude's answer from "worth about 0.7 K" to "not
worth doing at all". **That sensitivity is the result.** A single
break-even number quoted without the band would be a decision resting on
the midpoint of a rule of thumb.

## 3. Intervals, and which one matters (L8.3)

Three sources of uncertainty are attached to every scenario, because
they measure three different things and only the third is dominant:

| source | what it brackets | what it ignores |
|---|---|---|
| conformal band | one hour's cooling load, distribution-free | assumes the model's error distribution is unchanged by the intervention |
| week-block bootstrap | the annual mean of the difference | says nothing about the model being wrong |
| **parameter ensemble** | the same intervention under L6.8's equifinal parameter sets | everything outside the calibrated parameter space — plant assumptions, model form |

**The conformal band.** Normalised split conformal at α = 0.1 gives a
median half-width of **±1,030 kW on Claude (18.2% of predicted load)**
and **±502 kW on Luke (22.9%)**. Any hourly what-if figure carries that
band; it is the twin's own error against the meter, not the
intervention's.

**Coverage, measured three ways.** The gate item is marginal coverage
≥ 90% at α = 0.1. It is met — under the split whose assumption holds:

| split | Claude, constant | Claude, normalised | Luke, constant | Luke, normalised |
|---|---|---|---|---|
| interleaved weeks | **93.5% PASS** | 86.6% FAIL | **90.1% PASS** | **92.9% PASS** |
| contiguous forward | 99.2% PASS | 97.3% PASS | 86.3% FAIL | 86.4% FAIL |
| contiguous reversed | 49.9% FAIL | 68.9% FAIL | 92.8% PASS | 92.9% PASS |

The reversed row is the finding. Calibrating the interval on Claude's
last quarter and scoring it on the rest of the year gives **49.9%**
coverage from a method that guarantees 90% — because a quantile learned
in an Arizona winter is not a quantile that holds in an Arizona summer.
Conformal's guarantee is conditional on exchangeability and the seasons
break it. The interleaved split (alternating weeks, both sides spanning
every season) is the version whose assumption the data supports, and it
is the one the gate item is claimed under.

**Conditional coverage is worse than marginal everywhere.** The worst
month or load quintile under the interleaved split runs **54.5–85.6%**
against a 90% marginal number. Mondrian conformal — a separate quantile
per 5 K band of outdoor temperature, using M7's finding that the
residual's structure lives in temperature — narrows the gap (Claude's
worst conditional cell rises from 54.5% to 73.2%, and the median width
falls from 2,255 to 1,915 kW) without closing it.

**The Gaussian control is not a straw man and it wins on coverage.**
`mean ± 1.645σ` covers 95.4% on Claude and 94.9% on Luke under the
interleaved split — over-covering, with intervals 13% (Claude) and 19%
(Luke) wider than the constant-width conformal band, because a normal
fitted to a fat-tailed residual buys its coverage by being too wide
everywhere. Conformal is preferred here for landing closer to its stated
target, not for covering more.

## 4. The same results, said honestly (L8.4)

### Three overclaiming statements, rewritten

**❌ "The digital twin shows that raising the setpoint by 1 K saves 3.1%
of cooling energy."**

✅ "On the training year's weather, intervening on the calibrated model's
setpoint parameter by +1 K reduces predicted cooling load by 3.4% and
modelled plant electricity by 3.1% (bootstrap 90% CI −3.4% to −2.9%).
The load model is calibrated and passed ASHRAE G14 on a held-out year;
the electricity figure additionally depends on an uncalibrated generic
plant model, and the setpoint being intervened on is a fitted parameter
rather than a measured thermostat setting."

**❌ "Raising chilled-water supply temperature saves 4.2% — a validated
optimisation opportunity."**

✅ "Raising chilled-water supply by 2 K changes modelled plant
electricity by −4.2% if the coils can hold their design delta-T and by
+5.1% if the return temperature is unchanged. Which of those two
describes this building is not recorded in the dataset. Nothing here is
validated: no chiller sub-meter exists to validate it against, and no
intervention was ever performed."

**❌ "The model quantifies savings with 90% confidence intervals."**

✅ "Each scenario carries three intervals that bracket three different
things. The widest — the spread across parameter sets that fit the meter
equally well — reaches four times the point estimate on one scenario and
contains zero. None of the three covers the risk that the model's
structure is wrong, which M7 measured directly: 6.1% of Claude's
variance is unexplained by physics and ML together."

### What may be claimed from M8

- The twin produces interventional answers to questions with no
  observational counterpart in the data, and those answers differ from
  the correlational ones by up to 6× on the same building.
- The chiller/pump trade-off is quantified, with the assumption it turns
  on named and its break-even swept across a documented range.
- Every what-if result carries uncertainty from three named sources, and
  the dominant one is identified as parameter equifinality rather than
  sampling noise.
- Conformal intervals reach their stated coverage under a split whose
  assumption holds, and the two splits where they do not are reported
  with the mechanism.

### What may not be claimed

- **No saving is measured.** No setpoint was changed in any building. A
  counterfactual is a model output, and this project's advisory-mode
  framing (07_PROGRESS.md) applies unchanged.
- **No energy figure is validated.** The M6 gate validates cooling load,
  not electricity. `config/plant.yaml` is a set of stated assumptions
  from `03_DOMAIN_REFERENCE.md`, not measurements of this plant.
- **Nothing here transfers to another building** by argument. Two
  buildings gave opposite signs for one scenario.
- **The conformal coverage numbers are optimistic for an unseen year.**
  The physics parameters were fitted on all of 2016, so every scored
  hour was seen by the optimiser.

## Limitations

1. **The plant model is uncalibrated and unfalsifiable from this
   dataset.** Chiller curves, staging, pump sizing and tower approach
   are all traceable to `03_DOMAIN_REFERENCE.md` and none is traceable
   to Fox_education_Claude. The pump share in particular (8–12% band)
   moves Claude's chilled-water break-even from +0.70 K to below +0.5 K.
2. **`t_setpoint_c` is a fitted parameter, not a thermostat.** It landed
   at 24.83 °C on Claude and 24.26 °C on Luke, absorbing whatever else
   behaves like a temperature offset. Every zone-setpoint result is an
   intervention on the model, and its transfer to a real thermostat is
   an assumption.
3. **The curve set extrapolates.** With the chilled-water supply raised
   2 K, 1,334 of Claude's hours produced a COP above INV-1's ceiling of
   10 and were capped there, so the chiller's modelled saving is
   understated on 15% of the year. Curve inputs are already clamped to
   their documented validity box; the cap is what survives that.
4. **The equifinality ensemble has three members, on one building.**
   L6.8 found three behavioural parameter sets for Claude and none was
   run for Luke. A three-point range is a floor on the structural
   uncertainty, not an estimate of it.
5. **Interactions between scenarios are not modelled.** Each
   intervention is run alone against the same baseline. Raising the zone
   setpoint and the chilled-water supply together is not the sum of the
   two rows, because both move the part-load ratio the chiller curve is
   evaluated at.
6. **Hog_education_Cathleen is excluded.** Its model failed the G14 gate
   and ADR-015 attributes the failure to the inverse model's clip at
   zero. A counterfactual run on it would be a statement about that
   defect.

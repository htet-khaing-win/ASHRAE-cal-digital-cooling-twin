# Where the calibrated model's error lives — all three buildings, both years

> M7's first finding. The residual of the calibrated model is decomposed
> against five drivers, tested for curvature, and tested for randomness.
>
> **2017 is a RE-READ** of a test set opened once at L6.10 (ADR-002),
> logged in `07_PROGRESS.md`'s Test Set Access Log. No model, parameter
> or bound decision was taken from it — it answers only whether the
> shape measured on 2016 is also present on 2017. Nothing here can fit:
> the optimiser is not imported and parameters are read frozen from the
> 2016 calibration artifacts.
>
> Reproduce:
> `python scripts/analyse_residuals.py` (2016 only) and
> `python scripts/investigate_ushape.py --reread-test-set`
> Artifacts: `reports/calibration_runs/residuals_2016.json`,
> `reports/calibration_runs/ushape_investigation_2016_2017.json`
> Figures: `reports/figures/l7_1_residual_decomposition.png`,
> `l7_1c_ushape_both_years.png`, `l7_2_residual_acf.png`

## Summary

Three findings, in order of how much they change what M7 does next.

1. **Cathleen's error is almost pure curvature.** A straight line in
   outdoor temperature explains **0.3%** of its residual variance; a
   parabola explains **44.5%**. Identical on both years. No
   linear-in-ΔT model of any parameterisation can fit this building.
2. **Claude is not U-shaped** — this corrects the first reading of the
   binned profiles. Its cold turn-up carries 179 hours, 2.0% of the
   year. Claude is a hockey stick: flat below ~16 °C, **+54.4 kW/K**
   above it. That upper-arm slope is what L6.9's fold 2 measured
   (+51.50 kW/K) on a window lying entirely on the upper arm.
3. **No building's residual is white noise**, by a wide margin. Daily
   averaging leaves 34–80% of the variance standing against the 4.2% a
   white-noise residual would leave. The remaining error is systematic
   and therefore learnable.

The three buildings do **not** share one fault. A single model change
applied to all of them would help one and harm another.

> **§5 (added 2026-08-14) revises the cause of finding 1.** Cathleen's
> cold arm is not the balance-point/reheat term §2 hypothesised, and not
> a meter fault: it is the inverse model's clip at zero, binding at
> 4.95% of the year against a 5.00% allowance. Finding 1 itself stands —
> the fault is still one this model's form cannot express — but the term
> to add is a base load that survives cold weather, not a balance point.

## Method

`decompose_residual()` bins the residual against five drivers and reports,
per driver, the share of residual variance the binned means explain (η²)
beside the `(k−1)/(n−1)` a pure-noise residual would produce by chance.
The ratio of the two is the verdict; the raw η² alone is meaningless
because it rises with bin count.

Binning is on **predicted** load, never measured. Measurement noise sits
inside both the residual and the bin assignment, so an hour that reads
high reads high partly because its noise was positive — a model with
zero structural error scores a clean 377 kW slope that way
(`tests/test_residual.py` runs both versions on identical data).

`fit_residual_curvature()` makes two independent statements, because
either alone is arguable: a quadratic coefficient, and three
assumption-free band means. A U is claimed only when **both** arms
exceed the middle band by more than two combined standard errors **and**
the quadratic is positive. Bands are terciles of the **training year's**
driver, reused unchanged for the test year — recomputing them per year
would move the bands with the weather and compare two years across bands
that are not the same bands. Fixed physical thresholds were rejected:
`03_DOMAIN_REFERENCE.md` sets no balance-point temperature, and one
fixed edge cannot serve a portfolio running −24 °C to +44 °C.

Standard errors are inflated by the **effective sample size** (L7.2,
below). Uncorrected they are too small by 7–13× on these buildings.

## 1. Where the error lives (2016, structure ratio = η² ÷ its noise floor)

| Building | train CV(RMSE) | month | hour of day | predicted load | dry bulb | humidity |
|---|---|---|---|---|---|---|
| Fox_education_Claude | 11.72% | 353 | **8.5** | 430 | 203 | 323 |
| Bull_education_Luke | 14.13% | 74 | **12.2** | 28 | 12 | 87 |
| Hog_education_Cathleen | 28.66% | 333 | **2.2** | 697 | 258 | 608 |

Anything at or below ~3 is indistinguishable from chance.

**Mean residual is −0.0 kW on all three.** Annual NMBE is −0.00% on all
three. The objective drove the bias to zero and left the shape entirely
untouched — every headline NMBE in this repo should be read with this
table beside it.

**`hour_of_day` is clean everywhere.** This is a negative result worth
keeping. It is the driver the method is most sensitive to — on a
synthetic building with a withheld occupancy schedule it scores 191 —
and here it finds nothing. That is evidence the method does not
manufacture structure, and it independently re-confirms L6.7b's finding
that these buildings run 24/7.

**`predicted_load` is not a sixth finding.** Predicted load is largely a
function of temperature in this model, so that panel mirrors the
temperature panel. Marginal profiles localise an error; they do not
attribute it.

## 2. The curvature test — outdoor dry bulb, both years

| Building | Year | low band | mid band | high band | quadratic | R² line | R² quad | turn at | mass below | **U?** |
|---|---|---|---|---|---|---|---|---|---|---|
| Claude | 2016 | −395 | −221 | +617 | +2.75 | 0.234 | 0.330 | 16.2 °C | 15.4% | No |
| Claude | 2017 *rr* | −263 | +53 | +822 | +1.52 | 0.318 | 0.347 | 13.6 °C | 9.8% | No |
| Luke | 2016 | −15 | +29 | −20 | −0.16 | 0.000 | 0.001 | — | — | No |
| Luke | 2017 *rr* | +8 | +55 | +37 | +0.23 | 0.000 | 0.004 | — | — | No |
| **Cathleen** | **2016** | **+75** | **−279** | **+224** | **+1.22** | **0.003** | **0.445** | 8.6 °C | 41.6% | **YES** |
| **Cathleen** | **2017** *rr* | **+113** | **−312** | **+213** | **+1.36** | **0.003** | **0.481** | 11.1 °C | 47.8% | **YES** |

Band means in kW; positive means the model under-predicts. *rr* = re-read.

### Cathleen — the real finding

`R²(linear) = 0.003` on **both** years. A straight line through
Cathleen's residual explains three tenths of one percent of its
variance. A parabola explains 44.5% and 48.1%. The two years' binned
profiles are nearly superimposed.

The residual runs +467 kW at −18.9 °C, down to −403 kW at 8.6 °C, back
up to +307 kW at 30.9 °C — on a building whose mean load is 1,122 kW.
The arms are 40% of mean load.

This is the argument that matters, and it is algebraic rather than
statistical: the model's weather response is
`ṁ·cp·(T_out − T_in) + UA·A·(T_out − T_in)`, linear in ΔT with a single
sign. **A linear function has no interior minimum.** Cathleen's residual
has one, on both years. Therefore no parameter set of this structure
fits this building, and no amount of search or bound-widening changes
that. Same class of argument as L6.7b's "no admissible parameter set of
THIS structure gets near G14".

Cathleen's humidity profile is U-shaped too (R² line 0.110/0.033 against
quad 0.349/0.396), which is expected: on this site the driest hours are
the coldest hours.

**Two mechanisms produce this identical shape and only one of them is
about the model.** See Q10. (a) Physics — a balance-point term:
dehumidification reheat, a dedicated outdoor-air unit conditioning very
cold air, or simultaneous heating and cooling. (b) Metering — the
chilled-water meter reads something that is not this building's cooling
in deep winter. ADR-012 found that class of fault in Eagle's chilled
water and Q8 found it in Fox's hot water, so it is not remote. The
discriminating test is cheap and available: `config/buildings.yaml`
records a `steam` companion meter for this building. **Run it before
changing any model.**

### Claude — a hockey stick, not a U

The first reading of the binned profile called Claude U-shaped on the
strength of two visibly raised cold bins. The bin counts say otherwise:

| Outdoor dry bulb | residual | hours | cumulative |
|---|---|---|---|
| 3.8 °C | **+894 kW** | 29 | 0.3% |
| 6.5 °C | **+381 kW** | 150 | 2.0% |
| 8.7 °C | −45 kW | 248 | 4.9% |
| 16.2 °C | −557 kW | 758 | 24.0% |
| 43.3 °C | +916 kW | 96 | 100% |

The cold arm carries **179 hours, 2.0% of the year**. It cannot lift a
tercile band spanning everything below 20.6 °C, and the mass-weighted
verdict correctly reads `No`. Both readings are true and they answer
different questions, so the script now prints `turn at` and `mass
below %` beside the verdict rather than letting the eye pick.

Claude's real shape is flat-to-declining below ~16 °C, then a steep
monotone under-prediction:

```
upper-arm slope = (+916 − (−557)) / (43.3 − 16.2) = +54.4 kW/K
L6.9 fold 2     = +51.50 kW/K   (26 May – 7 Aug: entirely on the upper arm)
full-year linear = +28.06 kW/K  (diluted by the flat region below 16 °C)
```

Fold 2 was right and the full-year linear slope **understates** the
fault by roughly half.

The consequence is Q11. The deficit is concentrated above ~16 °C, and
below 16 °C the model already **over-predicts by up to 557 kW**. Both of
the model's weather terms are linear in ΔT, so raising either adds slope
everywhere and trades one error for another. `vent_flow_kg_per_s` is
already calibrated at 179.60 against its 180.0 ceiling (10 ACH,
fume-hood laboratory) and `ua_envelope_w_per_m2k` at its 3.0 ceiling —
ADR-011's ventilation term is implemented and the deficit is still
there. **The missing term must switch on with temperature or humidity
rather than scale with ΔT throughout.** First suspect: the latent
trigger, a fixed `supply_humidity_ratio` of 0.0092 that
`config/calibration.yaml` already records as known-imperfect in exactly
this direction.

### Luke — nothing to fix

R²(quad) 0.001 and 0.004. Luke's residual has essentially no temperature
structure, which is consistent with it being the best test-year
performer (11.14%). Applying Cathleen's or Claude's fix here would be
fitting noise.

## 3. Is the residual random? (L7.2)

| Building | Year | ρ(1) | ρ(24) | ρ(168) | Ljung-Box Q | p | daily variance share |
|---|---|---|---|---|---|---|---|
| Claude | 2016 | 0.826 | 0.598 | 0.557 | 347,467 | ~0 | 0.584 |
| Claude | 2017 *rr* | 0.864 | 0.800 | 0.703 | 747,126 | ~0 | 0.785 |
| Luke | 2016 | 0.590 | 0.302 | 0.155 | 46,365 | ~0 | 0.338 |
| Luke | 2017 *rr* | 0.675 | 0.362 | 0.196 | 52,339 | ~0 | 0.413 |
| Cathleen | 2016 | 0.966 | 0.701 | 0.483 | 428,369 | ~0 | 0.796 |
| Cathleen | 2017 *rr* | 0.961 | 0.667 | 0.422 | 259,447 | ~0 | 0.745 |

**Do not quote the p-value as the finding.** At n = 8,760 the Ljung-Box
test rejects white noise for autocorrelations far too small to matter,
so p is effectively zero for every building energy residual ever
measured. It is computed because its *absence* would be conspicuous, and
because a p that was **not** tiny would be genuinely surprising.

The decisive number is the last column, against a white-noise null of
1/24 = 0.042: **8× to 19×**. Averaging 24 independent draws cuts
variance 24-fold; an error that survives that is not measurement noise.
It is signal the model has not taken, and it is learnable — which is
what makes L7.3's hybrid residual model worth building.

## 4. n is not n

The autocorrelations above invalidate every standard error computed as
`s/√n`, including the ones in §2 as first written. Corrected via the
summed-autocorrelation effective sample size:

| Building | Year | n | n_eff | standard errors too small by |
|---|---|---|---|---|
| Claude | 2016 | 8,782 | 54 | **12.7×** |
| Claude | 2017 *rr* | 8,741 | 37 | **15.5×** |
| Luke | 2016 | 8,572 | 154 | 7.5× |
| Luke | 2017 *rr* | 8,607 | 160 | 7.3× |
| Cathleen | 2016 | 8,782 | 49 | **13.3×** |
| Cathleen | 2017 *rr* | 8,758 | 67 | 11.5× |

Every figure above is an **upper bound**: the autocorrelation was still
positive at the 168-lag truncation on all six series, so the true
effective sample sizes are smaller and the corrected standard errors are
still optimistic.

The AR(1) shortcut `n(1−ρ₁)/(1+ρ₁)` is not used. These residuals are
nothing like AR(1) — Claude's ρ(1) of 0.826 would imply ρ(24) = 0.010
and the measured value is 0.598 — and the shortcut errs in the direction
that matters, reporting a comfortably large effective sample size while
leaving the standard errors too small.

**Every verdict in §2 survives the correction.** Cathleen's arms clear
their corrected thresholds by 1.6× and 2.2× on 2016 and by 2.3× and 2.5×
on 2017; Claude's and Luke's low arms are negative and were never U
verdicts.

## What is established and what is not

**Established.** The verdicts, band means, R² values and diagnostics are
arithmetic over 8,500+ hours per building-year, reproduced on two years
with band edges fixed from the first. CV(RMSE) recomputed inside the
analysis reproduced the gate exactly (11.58 / 11.14 / 31.65), which is
the check that the residual analysed is the residual that was reported.

**Not established.** Every *mechanism* named here — balance point,
reheat, latent trigger — is a hypothesis consistent with a shape. BDG2
records no supply-air temperature, no damper position and no airflow
(Q8 states the same limit), so none can be confirmed from this dataset
alone. The metering-artifact alternative for Cathleen is untested.

**Weaknesses of this analysis.** 2017 is a re-read, so its agreement is
confirmatory but the year is no longer held out; the shape was measured
on 2016 first and predicted, which is the strongest form still
available, but it is not a pre-registered out-of-sample test. Cathleen's
2016 residual is not independent evidence — its parameters were fitted
on those hours — and only the 2017 agreement makes the finding
structural. Terciles are defensible but a choice; a different quantile
could move a marginal verdict, and the honest statement about Claude's
cold arm is "real and small", not "absent".

## 5. Q10 resolved — the cold arm is the model's own floor

> Added 2026-08-14. `python scripts/check_companion_meter.py`, training
> year only. Artifact `reports/calibration_runs/companion_meter_check_2016.json`,
> figure `reports/figures/l7_2_companion_meter.png`.

Q10 offered two explanations for Cathleen's rising cold arm: physics (a
balance-point or reheat term) or metering (a chilled-water reading that
is not this building's cooling in winter). **Neither is right.**

### The reheat hypothesis is dead

The raw correlation looks damning, and it is the trap:

| Building | meter | Pearson (cold band) | Spearman (cold band) | **matched-temperature effect** |
|---|---|---|---|---|
| Cathleen | steam | **+0.694** | **+0.772** | **−28.9 ± 51.6 kW** |
| Luke *(control)* | steam | +0.544 | +0.213 | +88.3 ± 58.2 kW |
| Claude | hotwater | +0.479 | +0.442 | −74.8 ± 162.7 kW |

A heating meter and a cold-weather residual both rise as it gets colder,
so they correlate whether or not one causes the other. Hold outdoor
temperature fixed — bin at 2.5 K, remove the within-bin temperature
trend, split each bin at the median steam reading — and the effect
vanishes. Hours when Cathleen is taking *more* steam do **not** have a
higher cooling residual at the same outdoor temperature.

**Luke is the control and it earns its place:** same steam meter, no
U-shaped residual, and a *larger* matched effect than Cathleen's. Any
reading that made Cathleen's steam special would have to explain that.

Claude's hotwater result also gives **no support** to Q8's reheat
hypothesis under this test — a stronger test than the raw +0.112
correlation `diagnose_crossval_fold.py` originally reported.

Magnitudes are never used anywhere here (Q8: companion-meter units are
not trusted). The split depends only on the meter's *order* within each
temperature bin, so a monotone unit error changes nothing.

### The metering hypothesis is dead

| Building | cold-band load CV | cold hours on annual floor | cold-band corr with T |
|---|---|---|---|
| Claude | 0.211 | 4.1% | 0.860 |
| Luke | 0.376 | 2.6% | 0.647 |
| Cathleen | 0.193 | 13.9% | **0.131** |

Cathleen's winter load is not flat (CV 0.193), does not pile up on the
year's floor, and was never flagged by M3's C3/C4 flatline rules. It is
not a stuck meter or a bypass reading.

But look at the last column. **In the cold band Cathleen's measured load
barely responds to outdoor temperature at all** — 0.131 against 0.860
and 0.647. It keeps a base cooling load that the weather does not drive.

### The actual cause: `inverse_cooling_load` clips at zero

| Building | clipped, year | allowed | clipped, cold band | meter read while clipped | binding? |
|---|---|---|---|---|---|
| Claude | 0.01% | 5.0% | 0.0% | — | No |
| Luke | 0.00% | 5.0% | 0.0% | — | No |
| **Cathleen** | **4.95%** | **5.0%** | **15.4%** | **449 kW** (CV 0.20) | **YES** |

Band by band:

| Outdoor dry bulb | hours | measured | predicted | residual | clipped |
|---|---|---|---|---|---|
| −30 to −15 °C | 286 | 444 kW | **0 kW** | **+444 kW** | **100%** |
| −15 to −5 °C | 785 | 473 | 168 | +305 | 19% |
| −5 to +5 °C | 2,044 | 479 | 594 | −115 | 0% |
| +5 to +15 °C | 1,974 | 668 | 1,032 | −364 | 0% |
| +15 to +25 °C | 2,796 | 1,762 | 1,625 | +137 | 0% |
| +25 to +40 °C | 897 | 2,375 | 2,149 | +226 | 0% |

The clip at zero is physically correct — a cooling coil cannot heat —
but it gives the model a hard floor that this building does not share.
Below −15 °C the model predicts **exactly zero for every hour** while
the meter reads a steady 444 kW. The residual there is
`measured − 0 = measured`: **arithmetic, not missing physics.**

And the fit sat *on* the constraint. L6.6's objective penalises clipping
beyond `max_clipped_fraction = 0.05`; Cathleen came in at **4.95%**.
That is a binding constraint, and it must be read exactly as Q7 taught
us to read a parameter pinned at its bound: **a statement about the
model's form, not about the threshold.** Nothing in M6 was reading it.

**Mechanism.** Cathleen keeps a weather-insensitive base cooling load.
The model's only constant term, `internal_gain_w_per_m2`, competes
against a large negative ΔT term in freezing weather, so the fit must
either raise the gain and break summer or let winter clip. It clipped.

**Consequence.** The change Cathleen needs is a **base-load term that
survives cold weather**, not the balance-point term Q10 hypothesised.
Both meters are exonerated. This is still a model-structure finding, and
the §2 conclusion stands — but the term to add is a different one.

## 6. The change-point baseline — the finding as a measurement (ADR-015)

> Added 2026-08-14. `python scripts/fit_change_point_baseline.py`,
> training year only. Artifact
> `reports/calibration_runs/change_point_baseline_2016.json`, figure
> `reports/figures/l7_2_change_point_baseline.png`.

§5 argued that the RC model's *form* cannot hold a base load through
winter. A three-parameter ASHRAE/IMT change-point model —
`load = base + slope · max(0, T − T_cp)` — is the standard inverse model
for exactly that shape. Fitting it as a **baseline** (L6.4's family, same
`ashrae_g14_pass()`, no change to the twin) turns the argument into a
number.

| Building | model | p | CV(RMSE) | G14 |
|---|---|---|---|---|
| Claude | **calibrated RC** | 5 | **11.72%** | PASS |
| | change-point | 3 | 14.20% | PASS |
| | linear regression | 2 | 14.76% | PASS |
| | annual mean | 1 | 40.75% | FAIL |
| Luke | **calibrated RC** | 5 | **14.13%** | PASS |
| | change-point | 3 | 20.53% | PASS |
| | linear regression | 2 | 20.88% | PASS |
| | annual mean | 1 | 40.11% | FAIL |
| **Cathleen** | calibrated RC | 5 | 28.66% | PASS |
| | **change-point** | **3** | **24.28%** | **PASS** |
| | linear regression | 2 | 38.85% | FAIL |
| | annual mean | 1 | 69.09% | FAIL |

| Building | T_cp | base | slope | RC | 3P | 3P beats RC by |
|---|---|---|---|---|---|---|
| Claude | 12.8 °C | 3,045 kW | 307.4 kW/K | 11.72% | 14.20% | **−21.1%** |
| Luke | 7.8 °C | 979 kW | 121.1 kW/K | 14.13% | 20.53% | **−45.3%** |
| **Cathleen** | **8.3 °C** | **480 kW** | 105.3 kW/K | 28.66% | 24.28% | **+15.3%** |

**On Claude and Luke the physics earns its extra parameters** — the RC
model beats the change-point curve by 21% and 45% relative. That is the
expected and desirable case, and it is what makes the third row mean
something.

**On Cathleen a three-parameter curve beats the five-parameter physics
model by 15.3% relative.** The gap is not calibration quality. It is the
model's form.

Two independent methods now agree on the same numbers. §5 derived a
~444 kW weather-insensitive base load and a residual minimum at
8.6 °C from the clipping analysis. The change-point fit, which knows
nothing about any of that, independently recovers **base = 480 kW** and
**T_cp = 8.3 °C**.

The figure makes the mechanism visible in one panel: from −24 °C to
about −8 °C the calibrated RC line sits **on zero** while both the meter
and the change-point model hold ~480 kW.

**What this does not mean.** The change-point model is not a better
twin — it is a better *curve*. It has no physical parameters to
intervene on, so it cannot answer a single counterfactual question,
which is the entire purpose of M8. It is a measuring instrument for the
RC model's form, and that is all it is used for here. ADR-015 authorised
it on that basis and declined to change the twin.

## 7. Q11 resolved — the latent term's trigger, not its size

> Added 2026-08-14. `python scripts/check_humidity_trigger.py`, training
> year only. Artifact `reports/calibration_runs/humidity_trigger_2016.json`,
> figure `reports/figures/l7_2_humidity_trigger.png`.

§2 left Claude with a deficit concentrated above ~16 °C that neither of
the model's weather terms can supply, because both are linear in ΔT and
raising either worsens the shoulder. Q11 named the first suspect: the
latent term's trigger.

```
latent_w = vent_flow · h_fg · max(w_outdoor − w_supply, 0)
                                              ^^^^^^^^ fixed at 0.0092 kg/kg
```

`w_supply` is a stated assumption (ADR-011, never fitted), so the latent
term is **exactly zero for every hour drier than 9.2 g/kg**.

### How much of the year the latent term is switched on at all

| Building | vent flow | outdoor w (p50) | **latent ON** | latent kW when on | **% of predicted load** |
|---|---|---|---|---|---|
| Claude | 179.6 kg/s | 4.8 g/kg | **16.8%** | 1,046 kW | **2.55%** |
| Luke | 56.4 kg/s | 12.2 g/kg | **66.0%** | 774 kW | **19.33%** |
| Cathleen | 40.9 kg/s | 5.0 g/kg | 25.7% | 298 kW | 6.82% |

On Claude — a Tempe laboratory with a median outdoor humidity of
4.8 g/kg — the term the calibration fitted a 179.6 kg/s flow for is
**inert 83% of the time** and contributes **2.55%** of predicted load.

### The test: at matched outdoor temperature, do humid hours run higher?

| Building | region | hours | **humid − dry** | ± | humid higher? |
|---|---|---|---|---|---|
| **Claude** | whole year | 8,782 | **+431.8** | 157.2 | **Yes** |
| | **upper arm (≥ 16.2 °C)** | 7,000 | **+553.3** | 191.1 | **Yes** |
| | lower region (< 16.2 °C) | 1,782 | −59.1 | 184.4 | No |
| Luke | whole year | 8,572 | +92.3 | 58.1 | **No** |
| Cathleen | whole year | 8,782 | +114.7 | 44.4 | Yes |
| | upper arm (≥ 8.6 °C) | 4,928 | **+191.8** | 73.1 | **Yes** |
| | lower region (< 8.6 °C) | 3,854 | +16.3 | 38.5 | No |

kW, after the within-bin temperature trend is removed; ± is the standard
error inflated by the effective sample size (L7.2).

**This is the signature Q11 pre-specified for a wrong TRIGGER:** the
effect is confined to the upper arm and absent below it. Had it been
present everywhere, the term would merely be mis-sized; had it been
absent everywhere, humidity would not be the missing term at all.

The residual is what remains **after** the latent term has been applied.
A humidity effect still standing at +553 kW on the hot hours means the
term as implemented is wrong, not missing.

### Luke is the control, and it closes the argument

| Building | latent term active | latent share of load | residual humidity effect |
|---|---|---|---|
| **Luke** | **66.0%** | **19.33%** | **none** (+92 ± 58) |
| Cathleen | 25.7% | 6.82% | +192 ± 73 |
| **Claude** | **16.8%** | **2.55%** | **+553 ± 191** |

The relationship is inverse and monotone across the portfolio: **the
less the latent term is active, the more humidity is left standing in
the residual.** On Luke — Austin, 66% of hours above the trigger — the
term does real work (19% of predicted load) and the residual carries no
humidity effect at all. On Claude the term is switched off for most of
the year and the effect is at its largest.

That is as close to a confirmed mechanism as this dataset permits. It is
still an inference: BDG2 records no supply-air state, so the true coil
condition is unknown, and `w_supply = 0.0092` remains an assumption
whose only defence is that it is stated rather than fitted.

**Consequence.** Claude's missing term is not more ventilation and not
more envelope UA — both are already at their ceilings and both would
worsen the shoulder. It is a latent term that engages where the building
actually dehumidifies. Any fix touches ADR-011's stated assumption,
which makes it a model-structure change and therefore subject to the
same ADR-015 reasoning: it would invalidate the gate for all three
buildings. **Not implemented. Recorded.**

> A caveat on Luke's rows: its residual minimum sits at 38.1 °C — the
> top of its range, because Luke has no hockey stick — so the upper-arm
> split had 13 hours and was correctly skipped rather than reported. Only
> the whole-year row is meaningful for Luke, and it is null.

## Next

1. ~~**Q10**~~ — resolved above.
2. ~~**ADR-015**~~ — recorded; the change-point baseline is §6.
3. ~~**Q11**~~ — resolved in §7. The upper arm IS carried by the humid
   half (+553 ± 191 kW), the lower region is not, and the latent term is
   inert for 83% of Claude's year.
4. Only then, L7.3 — **excluding Cathleen**, per ADR-015: an ML residual
   model fitted on her residual would learn to undo the clip at zero,
   which is patching arithmetic, not learning physics.

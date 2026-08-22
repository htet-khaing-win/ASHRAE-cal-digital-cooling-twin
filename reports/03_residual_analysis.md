# One model structure, three climates — where the humidity term breaks

> M7's closing report, and the project's research contribution. The
> calibrated model carries ADR-011's ventilation term, whose latent part
> is a **threshold**, not a scale factor:
>
> ```
> latent_kw = vent_flow * h_fg * max(w_outdoor - w_supply, 0)
> ```
>
> `w_supply` is a STATED ASSUMPTION fixed at 9.2 g/kg and never fitted.
> This report measures what that single number does at three sites whose
> humid-hour share differs by a factor of nine, and sweeps it to find out
> whether the value is defensible at any of them.
>
> **Training year only (ADR-002).** Calibrated parameters are read frozen
> from the 2016 artifacts; the optimiser is not imported. The sweep
> changes a stated assumption, never a fitted value, and **no value found
> here is adopted** — adopting one would fall under ADR-015.
>
> Reproduce: `python scripts/compare_site_humidity.py`
> Artifact: `reports/calibration_runs/site_humidity_2016.json`
> Figure: `reports/figures/l7_5_humid_vs_dry.png`
>
> Companion reports: `06_residual_curvature.md` (where the error lives),
> `07_explainability.md` (what an importance plot can and cannot say).

## Summary

1. **The same assumption is a different model at each site.** At
   9.2 g/kg the latent term is switched on for **16.8%** of Claude's
   year, **25.7%** of Cathleen's and **66.0%** of Luke's, carrying
   **2.5% / 6.8% / 19.3%** of the predicted load. Nobody changed a line
   of code; the climate changed the model.
2. **The two dry sites under-predict their humid hours badly, and the
   humid site does not.** Physics NMBE on hours above 12 g/kg:
   Claude **+9.40%**, Cathleen **+14.66%**, Luke **+0.94%** — against
   +0.44%, −5.10% and +0.58% on their own dry hours. On the two
   buildings that passed the gate the pattern is the same and has a
   mechanism: **the calibration objective is dominated by whichever
   regime holds most of the hours.** Claude's humid hours are 7.1% of
   the scored year and the fit abandons them; Luke's are 60.1% and the
   fit is built on them.
3. **The stated 9.2 g/kg is near-optimal at the humid site and wrong at
   both dry ones.** Sweeping the trigger with parameters frozen, the
   CV(RMSE) minimum sits at **9.2 g/kg for Luke** (14.13%, the stated
   value itself), at **8.0 g/kg for Claude** (10.54% against 11.72%,
   −1.18 pp) and at **8.0 g/kg for Cathleen** (26.61% against 28.66%,
   −2.05 pp). The humidity signal the term fails to remove crosses zero
   at roughly **7 g/kg on Claude, 6.5 on Luke and 4 on Cathleen** — in
   every case **below** the stated assumption.

The claim this licenses is about the METHOD, not about climates: a
threshold assumption inside a physical term must be validated per
climate, because its reach — the share of hours it touches at all — is
set by the weather and not by the modeller. It is not a claim that
grey-box models work better in humid climates; see Limitations.

## The three sites

`humid hours` counts hours above `select.HUMID_HUMIDITY_RATIO`
(12 g/kg), the same threshold M2 used when the buildings were chosen.

| building | site | humid hours | outdoor w, median | vent flow, kg/s | whole-year CV(RMSE) |
|---|---|---:|---:|---:|---:|
| Fox_education_Claude | Fox | **5.8%** | 4.8 g/kg | 179.6 | 11.72% |
| Hog_education_Cathleen | Hog | 11.3% | 5.0 g/kg | 40.9 | 28.66% ⚠ |
| Bull_education_Luke | Bull | **50.9%** | 12.2 g/kg | 56.4 | 14.13% |

⚠ Cathleen's CV(RMSE) is **not interpretable as a humidity result**: its
error is dominated by the clip-at-zero defect of ADR-015. It is carried
here because a structure claim tested on two buildings when a third is
available is weaker for no reason, and its humidity diagnostics are
sound even where its accuracy is not. It has no hybrid arm.

## Both model structures, by regime

Scored on the hours the out-of-fold correction covers, so the physics
and the hybrid are compared on one set of hours. `ML %` is measured
against **that regime's own variance**, so the two columns are
comparable in sign and not in magnitude.

### Fox_education_Claude — dry site

| regime | hours | mean load | physics NMBE | physics CV | hybrid CV | ML % |
|---|---:|---:|---:|---:|---:|---:|
| dry (w ≤ 12) | 6,016 | 7,137 kW | +0.44% | 10.66% | 8.69% | 2.92% |
| humid (w > 12) | 463 | 11,187 kW | **+9.40%** | 13.67% | 10.57% | **38.10%** |

The humid hours carry a mean residual of **+1,040 kW** — the model
under-predicts them — on 7.1% of the scored year. The learnt correction
recovers 38% of the variance inside that regime, which is the largest
ML share anywhere in this project and is the clearest statement of what
the physics is missing: a term that only matters when the outdoor air is
wet, on a site where that is rare.

### Bull_education_Luke — humid site

| regime | hours | mean load | physics NMBE | physics CV | hybrid CV | ML % |
|---|---:|---:|---:|---:|---:|---:|
| dry (w ≤ 12) | 2,512 | 1,906 kW | +0.58% | 18.11% | 16.14% | 6.55% |
| humid (w > 12) | 3,792 | 3,505 kW | +0.94% | **10.23%** | 10.20% | 0.23% |

The mirror image. Luke's humid hours are the majority and the model fits
them well; its minority dry hours are where the error and the entire
learnt correction live.

### Hog_education_Cathleen — negative case

| regime | hours | mean load | physics NMBE | physics CV |
|---|---:|---:|---:|---:|
| dry (w ≤ 12) | 7,794 | 937 kW | −5.10% | 32.11% |
| humid (w > 12) | 988 | 2,583 kW | **+14.66%** | 17.59% |

Opposite-signed bias between regimes behind an annual NMBE of −0.00% —
the same signature ADR-014 recorded seasonally. Note that Cathleen
**breaks the majority-regime pattern**: its minority humid hours fit
better than its majority dry hours. That is consistent with ADR-015 (the
dry-hour error is the clip at zero, which has nothing to do with
humidity) and it is reported rather than dropped.

## The trigger sweep

Calibrated parameters frozen; only `w_supply` moves. **Bold** marks the
stated assumption; the CV(RMSE) minimum of each row block is italicised.

| w_supply | Claude active | Claude CV | Claude signal | Luke active | Luke CV | Luke signal | Cathleen active | Cathleen CV | Cathleen signal |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.0 | 61.6% | 16.21% | −804 ± 175 | 89.9% | 26.04% | −66 ± 56 | 58.4% | 33.61% | +6 ± 46 |
| 6.0 | 36.4% | 11.02% | −221 ± 158 | 80.0% | 18.66% | −14 ± 56 | 43.1% | 26.94% | +44 ± 44 |
| 8.0 | 22.7% | *10.54%* | +226 ± 152 | 71.0% | 14.36% | +50 ± 57 | 31.8% | *26.61%* | +87 ± 43 |
| **9.2** | **16.8%** | **11.72%** | **+432 ± 157** | **66.0%** | ***14.13%*** | **+92 ± 58** | **25.7%** | **28.66%** | **+115 ± 44** |
| 12.0 | 5.8% | 14.66% | +684 ± 187 | 50.9% | 18.80% | +205 ± 63 | 11.3% | 34.73% | +189 ± 48 |
| 16.0 | 0.5% | 16.57% | +780 ± 219 | 28.4% | 27.77% | +383 ± 68 | 1.4% | 38.47% | +230 ± 51 |

`signal` is the humid-hour residual excess at **matched outdoor
temperature** (L7.2's matched split: bin on temperature, remove the
within-bin temperature trend, split each bin at its own median
humidity), ± 1 SEM already inflated by each building's effective sample
ratio (0.0062 / 0.0179 / 0.0056 — hourly residuals this autocorrelated
are worth far fewer independent observations than their count).

Two things to read from it.

- **The signal is monotone in the trigger on all three buildings**, and
  its zero lies below 9.2 g/kg everywhere. A lower coil humidity ratio
  — a colder coil, or the recognition that the supply air is not the
  100% outdoor air the model assumes — would remove humidity from the
  residual on every building tested.
- **The two criteria disagree.** The trigger that zeroes the humidity
  signal (~7 g/kg on Claude) is *below* the one that minimises CV(RMSE)
  (8.0 g/kg). They are not the same question: one removes a conditional
  bias, the other minimises total error against a flow that was fitted
  at 9.2.

## Why the sweep is not a recommendation

`vent_flow_kg_per_s` was calibrated **with the trigger at 9.2 g/kg**,
and the optimiser drove annual NMBE to −0.00% there. Every other column
in the sweep therefore inherits a flow chosen for a different threshold:
lowering the trigger adds latent load in hours between the new and old
values, which is why NMBE runs to −11.32% at 4.0 g/kg on Claude. A
proper answer requires a **joint refit of (trigger, flow)**, which is a
re-calibration, changes ADR-011's stated assumption, and falls under
ADR-015. What the sweep establishes is the SHAPE and the SIGN — that the
residual's humidity signal responds monotonically to a number nobody
fitted, and that its zero is on the same side of the assumption at all
three sites. That is evidence for a future ADR, not a parameter value.

## What may be claimed

- ✅ "A fixed supply-humidity threshold inside the latent term reaches
  16.8% of the hours at the dry site and 66.0% at the humid one, so the
  same model structure is a different model at each."
- ✅ "On the dry site the model under-predicts its humid hours by 9.40%
  NMBE while the humid site under-predicts by 0.94%; the learnt
  correction recovers 38% of the variance in that regime."
- ✅ "Sweeping the assumption with parameters frozen moves the residual's
  humidity signal monotonically and its zero lies below 9.2 g/kg on all
  three buildings."
- ❌ "Grey-box calibration works better in humid climates." One building
  per climate group. This is a case study.
- ❌ "The supply humidity ratio should be 8 g/kg." The flow was fitted at
  9.2; a joint refit is required before any value is recommended.
- ❌ "The humidity term is the cause of Cathleen's failure." Its failure
  is the clip at zero (ADR-015), measured separately.

## Limitations

1. **n = 1 per climate group.** Site-level statements are case studies.
   The hour-level regime split is the part with statistical weight —
   463 to 7,794 hours per cell — and every hour-level statement above is
   *within* a building, where the site is held fixed by construction.
2. **Regime and load are confounded.** Humid hours are also hot,
   high-load hours (Claude 11,187 kW against 7,137 kW; Luke 3,505
   against 1,906). NMBE is a relative measure so it is not confounded by
   scale, but CV(RMSE) between regimes partly reflects that the humid
   regime has a larger mean to normalise by. The matched split in the
   sweep is the version that holds temperature fixed; the regime table
   does not.
3. **Both dry sites are dry in the same way.** Fox and Hog have almost
   identical median outdoor humidity (4.8 and 5.0 g/kg), so the "two
   dry buildings agree" observation is weaker than two independent
   climates would be.
4. **The 12 g/kg regime threshold and the 9.2 g/kg trigger are
   different numbers with different origins** — the first is M2's
   site-screening definition, the second is ADR-011's coil assumption.
   They are deliberately not reconciled: aligning them would make the
   regime split a tautology of the term being tested.
5. **Training year only.** Nothing here has been checked against 2017,
   and under ADR-002 it will not be before M9.

# Building selection — the ADR-012 screen

*Generated 2026-08-13. Screen run on 2016 only; 2017 remains the held-out
test set (ADR-002).*

## Why the selection was redone mid-project

The original selection (L2.2) scored buildings on data **completeness** —
missing-hours fraction, meter availability, metadata presence. It never asked
whether the load is **explainable** by the drivers a cooling twin actually has.

Those two properties turn out to be nearly unrelated. `Fox_education_Theodore`
had 0.0% missing data and was the top-ranked candidate; it also sits at the
49th percentile of weather-explainability, and M6 stalled there at CV(RMSE)
42.25% against a 30% gate. The ceiling analysis showed no model of any
structure could have passed on that building-year.

## What the screen measures

| Metric | Definition | Threshold |
|---|---|---|
| Weather-explainability ceiling | Held-out CV(RMSE) of the conditional mean of load on binned (temperature, humidity). The best **any** model on those drivers could reach. | ≤ 30% (the G14 hourly limit) |
| Operational stability | Largest 11-day excursion from the weather expectation, ÷ the median day's load | ≤ 0.5 |
| Cooling intensity | Mean cooling ÷ floor area | 5–1,000 W/m² |
| Meter scope | Annual chilled water ÷ the building's own electricity | ≤ 10.0 |
| M3 cleaning | Fraction removed by the C1–C9 pipeline | ≤ 15% |

Every threshold and its anchor is recorded in ADR-012. The screen runs **after**
the M3 quality rules, never instead of them — a stuck sensor is trivially
predictable and scores an excellent ceiling (see the exclusions below).

## Selected

| Role | Building | Site (climate) | Sub-use | Ceiling | Stability | Humid hours |
|---|---|---|---|---|---|---|
| Primary | `Fox_education_Claude` | Fox — Tempe AZ (dry) | College Laboratory | 10.0% | 0.19 | 6% |
| Generalisation | `Bull_education_Luke` | Bull — Austin TX (humid) | Student Union | 13.7% | 0.10 | 51% |
| Generalisation | `Hog_education_Cathleen` | Hog — Minneapolis MN | College Laboratory | 17.4% | 0.25 | 11% |
| **Negative case** | `Fox_education_Theodore` | Fox — Tempe AZ (dry) | College Laboratory | **38.3%** ✗ | **1.66** ✗ | 6% |

Three distinct sites, and both a dry site and a humid site, so the humid-vs-dry
comparison (L7.5) and the latent term (ADR-011) can both be demonstrated.

## Retained negative case

`Fox_education_Theodore` fails the screen on both the ceiling and the stability
criteria and is kept in `config/buildings.yaml` anyway, with its rejection
reasons recorded.

This is the honesty mechanism for a mid-project re-selection. Choosing new
buildings after the old one failed is selection-on-outcome, and the standard
critique — *"you calibrated the buildings that were easy to calibrate"* — is
unanswerable if the failure quietly disappears from the repository. Theodore's
findings (Q6's laboratory identification, Q7's structural verdict, the
unexplained 11-day September 2016 load doubling) remain part of the project's
evidence.

## Exclusions worth reading

| Excluded | Why | What it taught |
|---|---|---|
| Eagle site (~90 buildings) | Median cooling intensity **24,306 W/m²**, ~400× anything physical, across every building at the site | CV(RMSE) is **scale-invariant** — a unit error is exactly as "explainable" as a correct meter. The screen needed a physical intensity check. |
| `Fox_education_Gloria` | M3 cleaning removes 38.25%: the meter is stuck at 2315.8281 kW for runs of up to **503 hours** across eight months | A flatlined meter is trivially predictable and ranked **first** on the raw screen. The quality rules must run *before* the screen, not after. |
| `Bull_office_Anne` | Cooling/electricity ratio **15.08** with an ordinary 13.8 W/m² electricity draw | A meter serving more than its building makes floor area the wrong normaliser for every per-area parameter. |
| Original stability metric | Rejected `Fox_education_Claude` (raw 1.84) and `Hog_public_Brad` (1.69) purely for sitting in hot climates | Measured on raw load, the statistic conflated seasonal swing with operational excursion. Measured on the **residual from the weather expectation**, Theodore scores 1.66 and everything else 0.09–0.30. |

Three of those four were defects in the screen itself, found by running it.

## Scale of the screen

```
1,636 metadata rows
  ->   271 pass the hard filters (completeness, meters, intensity, meter scope)
  ->   182 have enough usable hours after the weather join to be screened
  ->    54 survive the screen
  ->     3 selected (distinct sites) + 1 retained negative case
```

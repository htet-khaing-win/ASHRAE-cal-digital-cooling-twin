# Simultaneous heating and cooling in Fox laboratories (2016)

*Generated 2026-08-13. Exploration on the training year only. Opens Q8 in
`07_PROGRESS.md`.*

## Why this was investigated

Calibration needs a large constant term on every building tested — the cooling
load that survives the coldest hours runs 2.2–3.5× each building's entire
electricity intensity, so it cannot be equipment gains from metered power. The
question is what produces it. A terminal-reheat system is the standard
candidate: air is cooled to a fixed supply condition and then reheated at the
zone, so the cooling coil runs regardless of the weather.

Four Fox College Laboratories were compared: the new primary (`Claude`), the
retained negative case (`Theodore`), and the two nearest laboratory peers by
cooling intensity (`Stacia` 372.0 W/m², `Yolande` 321.8 W/m², against Claude's
383.4 W/m²).

## ⚠️ The Fox hot-water meter is not in kW

`Fox_education_Claude` reports a mean hot water of 57,729 — 3,215 W/m². That is
not a plausible heating load for a building in Tempe. The Fox hot-water series
is therefore treated here as an **unscaled index**: only timing, overlap and
correlation are used, never magnitude. The same unit problem was found in the
Eagle site's chilled water (ADR-012), so it is not unique to this meter.

## Overlap and correlation

| Building | Hours | Both meters > 0 | Overlap | corr(hourly) | corr(daily) | corr(hot water, outdoor T) |
|---|---|---|---|---|---|---|
| `Fox_education_Claude` | 8,755 | 8,715 | **99.5%** | −0.676 | −0.804 | −0.750 |
| `Fox_education_Yolande` | 8,733 | 8,703 | **99.7%** | −0.431 | −0.731 | −0.444 |
| `Fox_education_Theodore` | 8,782 | 4,541 | 51.7% | −0.483 | −0.497 | −0.678 |
| `Fox_education_Stacia` | 8,782 | 2,775 | 31.6% | +0.480 | +0.561 | −0.458 |

**Two distinct operating regimes appear among laboratories at one site.**

## Monthly profiles (each meter normalised to its own annual mean)

```
Fox_education_Claude          Jan   Apr   Jul   Oct   Dec     outdoor T Jul = 36.4 C
  chilled water              0.50  0.85  1.56  1.06  0.57
  hot water                  1.38  0.95  0.79  0.89  1.31     <- never switches off

Fox_education_Yolande
  chilled water              0.30  0.82  1.81  1.03  0.34
  hot water                  1.23  1.09  0.78  0.86  1.09     <- never switches off

Fox_education_Theodore
  chilled water              0.60  0.69  1.23  1.38  0.56
  hot water                  2.64  0.70  0.00  0.16  4.00     <- seasonal, off Jun-Oct

Fox_education_Stacia
  chilled water              1.41  1.05  0.95  0.92  0.84
  hot water                  4.48  2.83  0.00  0.00  0.00     <- seasonal, off Jun-Dec
```

`Claude` and `Yolande` run heating **and** cooling in July at 36 °C outdoor,
every hour of the year. Their hot water dips only ~20% between January and
July. `Theodore` and `Stacia` show ordinary seasonal opposition, with heating
fully off through the cooling season.

## What this suggests

For `Claude` and `Yolande` the signature is consistent with constant-volume
terminal reheat, or a dew-point-controlled 100% outside-air laboratory system:
the coil cools continuously to a fixed supply condition and the terminal
reheats. That mechanism produces exactly the constant cooling floor measured
independently (Claude 166–178 W/m² in the coldest temperature bins), and it
explains why the calibrated `internal_gain_w_per_m2` of 372 W/m² is seven times
the building's electricity intensity — the parameter is absorbing coil load,
not internal gains.

**Not confirmed.** BDG2 records no supply-air temperature, no damper position
and no airflow, and the hot-water meter's units are wrong. This is a hypothesis
consistent with four independent observations, not a diagnosis.

## Correction to an earlier statement

The L6.7b analysis reported that Theodore's unexplained 11-day September 2016
load doubling was "consistent with simultaneous heating and cooling", on the
basis of window-mean hot water (0.0 before → 48.8 during → 396.4 after). At
**daily** resolution that reading does not hold:

```
                        chilled water                     hot water
                Claude  Theodore  Stacia  Yolande     Claude  Theodore
2016-09-11       9,329     3,289   1,114    7,830     44,842       0.0
2016-09-12       9,378     4,841   1,115    7,927     45,164       0.0   <- event starts
2016-09-19       8,384     6,655   1,114    7,015     45,522       0.0   <- event peak
2016-09-22      10,201     5,339   1,175    9,669     45,343       0.0
2016-09-23       7,360     2,884   1,098    5,994     58,334     179.0   <- event ends
```

Theodore's hot water is **zero throughout the event** — a single 537 reading on
16 September aside — and only begins after the event ends, which is the heating
season starting. The event is also Theodore-specific: Claude and Yolande's
cooling *falls* over the same days while Theodore's doubles.

So Theodore's September event remains **unexplained**. The reheat hypothesis
applies to Claude and Yolande's year-round baseline, not to Theodore's
excursion. Q8 carries both questions separately.

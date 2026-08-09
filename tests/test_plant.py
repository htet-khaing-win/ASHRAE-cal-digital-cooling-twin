"""Unit tests for the latent-load plant model
(src/cooling_twin/models/plant.py).

Test pattern 1 of 4 (05_ENGINEERING_STANDARDS.md SS3): known-answer
tests hand-computed from 03_DOMAIN_REFERENCE.md SS3's formula, plus
input-validation tests for every ValueError branch (cooling_load()'s
heating/humidification rejections, and INV-2's swapped-temperature
rejection -- the gap 06_ASSESSMENT.md's M5 gate flagged as untested).
"""

from __future__ import annotations

import pytest

from cooling_twin.models.plant import CoolingLoad, cooling_load, validate_chw_delta_t


def test_cooling_load_known_answer_sensible_only() -> None:
    """m_air=1 kg/s, T 10degC->0degC, no humidity change: Q_sensible =
    1 * 1006 * 10 = 10060 W = 10.06 kW exactly. Q_latent must be zero.
    """
    load = cooling_load(
        m_air_kg_per_s=1.0,
        t_entering_c=10.0,
        t_leaving_c=0.0,
        w_entering_kg_per_kg=0.008,
        w_leaving_kg_per_kg=0.008,
    )
    assert load.sensible_kw == pytest.approx(10.06)
    assert load.latent_kw == pytest.approx(0.0)
    assert load.total_kw == pytest.approx(10.06)


def test_cooling_load_known_answer_with_latent() -> None:
    """m_air=1 kg/s, no temperature change, humidity ratio drops by
    0.001 kg/kg: Q_latent = 1 * 2,450,000 * 0.001 = 2450 W = 2.45 kW.
    """
    load = cooling_load(
        m_air_kg_per_s=1.0,
        t_entering_c=13.0,
        t_leaving_c=13.0,
        w_entering_kg_per_kg=0.009,
        w_leaving_kg_per_kg=0.008,
    )
    assert load.sensible_kw == pytest.approx(0.0)
    assert load.latent_kw == pytest.approx(2.45)


def test_cooling_load_total_kw_is_the_sum() -> None:
    load = CoolingLoad(sensible_kw=100.0, latent_kw=40.0)
    assert load.total_kw == pytest.approx(140.0)


def test_cooling_load_scales_linearly_with_mass_flow() -> None:
    load_1x = cooling_load(1.0, 24.0, 13.0, 0.010, 0.008)
    load_2x = cooling_load(2.0, 24.0, 13.0, 0.010, 0.008)
    assert load_2x.total_kw == pytest.approx(2 * load_1x.total_kw)


def test_cooling_load_rejects_non_positive_mass_flow() -> None:
    with pytest.raises(ValueError, match="m_air_kg_per_s"):
        cooling_load(0.0, 24.0, 13.0, 0.010, 0.008)


def test_cooling_load_rejects_heating() -> None:
    """t_leaving_c > t_entering_c describes a heating coil, not cooling."""
    with pytest.raises(ValueError, match="heating"):
        cooling_load(1.0, t_entering_c=13.0, t_leaving_c=24.0, w_entering_kg_per_kg=0.008,
                      w_leaving_kg_per_kg=0.008)


def test_cooling_load_rejects_humidification() -> None:
    """w_leaving > w_entering describes a humidifying coil, not dehumidifying."""
    with pytest.raises(ValueError, match="humidification"):
        cooling_load(1.0, t_entering_c=24.0, t_leaving_c=13.0, w_entering_kg_per_kg=0.008,
                      w_leaving_kg_per_kg=0.010)


def test_validate_chw_delta_t_accepts_valid_pair() -> None:
    validate_chw_delta_t(t_chw_supply_c=6.7, t_chw_return_c=12.7)  # must not raise


def test_validate_chw_delta_t_rejects_equal_temps() -> None:
    with pytest.raises(ValueError, match="INV-2"):
        validate_chw_delta_t(t_chw_supply_c=7.0, t_chw_return_c=7.0)


def test_validate_chw_delta_t_rejects_swapped_pair() -> None:
    with pytest.raises(ValueError, match="INV-2"):
        validate_chw_delta_t(t_chw_supply_c=12.7, t_chw_return_c=6.7)

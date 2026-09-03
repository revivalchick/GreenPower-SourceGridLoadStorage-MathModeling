from pathlib import Path

import pytest

from green_energy.data import load_config, load_input_data


ROOT = Path(__file__).resolve().parents[1]


def test_competition_data_dimensions_and_totals():
    config = load_config(ROOT / "configs" / "base.yaml")
    data = load_input_data(ROOT / "data" / "processed", config)
    assert len(data.scenarios) == 24
    assert len(data.hours) == 24
    assert data.load_mw.sum() == pytest.approx(60.72, abs=1e-6)
    assert (data.typical.wind_mw.sum() + data.typical.pv_mw.sum()) == pytest.approx(
        603.448, abs=1e-6
    )


def test_hourly_prices_follow_problem_definition():
    config = load_config(ROOT / "configs" / "base.yaml")
    data = load_input_data(ROOT / "data" / "processed", config)
    assert data.price_yuan_per_kwh[0] == pytest.approx(0.3424)
    assert data.price_yuan_per_kwh[10] == pytest.approx(0.8024)
    assert data.price_yuan_per_kwh[15] == pytest.approx(0.6074)


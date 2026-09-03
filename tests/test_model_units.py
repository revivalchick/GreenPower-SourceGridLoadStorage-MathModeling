from pathlib import Path

import pytest

from green_energy.data import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_electrolyzer_unit_conversion_matches_problem_statement():
    config = load_config(ROOT / "configs" / "base.yaml")
    alk = config["equipment"]["ALK"]
    pem = config["equipment"]["PEM"]
    alk_10mw_kgph = 10.0 * 1000.0 * alk["efficiency"] / alk["base_specific_energy_kwh_per_kg_h2"]
    pem_10mw_kgph = 10.0 * 1000.0 * pem["efficiency"] / pem["base_specific_energy_kwh_per_kg_h2"]
    assert alk_10mw_kgph == pytest.approx(140.0)
    assert pem_10mw_kgph == pytest.approx(160.0)


def test_full_ammonia_line_requires_full_hydrogen_output():
    config = load_config(ROOT / "configs" / "base.yaml")
    alk = config["equipment"]["ALK"]
    pem = config["equipment"]["PEM"]
    nh3 = config["equipment"]["NH3"]
    h2_supply = sum(
        spec["rated_power_mw"]
        * 1000.0
        * spec["efficiency"]
        / spec["base_specific_energy_kwh_per_kg_h2"]
        for spec in (alk, pem)
    )
    nh3_full_kgph = nh3["rated_power_mw"] * 1000.0 / nh3["electricity_kwh_per_kg_nh3"]
    assert h2_supply == pytest.approx(600.0)
    assert h2_supply == pytest.approx(nh3_full_kgph * nh3["hydrogen_kg_per_kg_nh3"])


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Scenario:
    name: str
    wind_mw: np.ndarray
    pv_mw: np.ndarray


@dataclass(frozen=True)
class InputData:
    hours: np.ndarray
    load_mw: np.ndarray
    price_yuan_per_kwh: np.ndarray
    typical: Scenario
    scenarios: Dict[str, Scenario]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/prepare_data.py or keep the committed processed CSV files."
        )
    return pd.read_csv(path)


def load_input_data(data_dir: str | Path, config: dict) -> InputData:
    data_dir = Path(data_dir)
    load = _read_csv(data_dir / "load.csv")
    typical = _read_csv(data_dir / "typical.csv")
    wind_scenarios = _read_csv(data_dir / "wind_scenarios.csv")
    pv_scenarios = _read_csv(data_dir / "pv_scenarios.csv")
    prices = _read_csv(data_dir / "prices.csv")

    expected_hours = np.arange(24)
    for frame, name in [
        (load, "load"),
        (typical, "typical"),
        (wind_scenarios, "wind_scenarios"),
        (pv_scenarios, "pv_scenarios"),
        (prices, "prices"),
    ]:
        if len(frame) != 24 or not np.array_equal(frame["hour"].to_numpy(), expected_hours):
            raise ValueError(f"{name} must contain exactly hours 0..23")

    wind_capacity = float(config["renewables"]["wind_capacity_mw"])
    pv_capacity = float(config["renewables"]["pv_capacity_mw"])

    scenario_map: Dict[str, Scenario] = {}
    wind_cols = [col for col in wind_scenarios.columns if col.startswith("wind_")]
    pv_cols = [col for col in pv_scenarios.columns if col.startswith("pv_")]
    for wind_col in wind_cols:
        for pv_col in pv_cols:
            name = f"W{wind_col.split('_')[-1]}_P{pv_col.split('_')[-1]}"
            scenario_map[name] = Scenario(
                name=name,
                wind_mw=wind_scenarios[wind_col].to_numpy(dtype=float) * wind_capacity,
                pv_mw=pv_scenarios[pv_col].to_numpy(dtype=float) * pv_capacity,
            )

    if len(scenario_map) != 24:
        raise ValueError(f"Expected 24 wind-PV combinations, found {len(scenario_map)}")

    typical_scenario = Scenario(
        name="typical",
        wind_mw=typical["wind_pu"].to_numpy(dtype=float) * wind_capacity,
        pv_mw=typical["pv_pu"].to_numpy(dtype=float) * pv_capacity,
    )
    return InputData(
        hours=expected_hours,
        load_mw=load["load_pu"].to_numpy(dtype=float) * 6.0,
        price_yuan_per_kwh=prices["buy_price_yuan_per_kwh"].to_numpy(dtype=float),
        typical=typical_scenario,
        scenarios=scenario_map,
    )


def mean_scenario(scenarios: Dict[str, Scenario]) -> Scenario:
    ordered = list(scenarios.values())
    return Scenario(
        name="expected_profile",
        wind_mw=np.mean([item.wind_mw for item in ordered], axis=0),
        pv_mw=np.mean([item.pv_mw for item in ordered], axis=0),
    )


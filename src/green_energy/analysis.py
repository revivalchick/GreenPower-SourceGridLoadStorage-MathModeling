from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .data import InputData, Scenario


def extract_dispatch(model: pyo.ConcreteModel, input_data: InputData) -> pd.DataFrame:
    rows: list[dict] = []
    for s in model.S:
        for t in model.T:
            rows.append(
                {
                    "scenario": str(s),
                    "hour": int(t),
                    "load_mw": float(pyo.value(model.base_load[s, t])),
                    "wind_available_mw": float(pyo.value(model.wind_available[s, t])),
                    "pv_available_mw": float(pyo.value(model.pv_available[s, t])),
                    "wind_dispatch_mw": float(pyo.value(model.wind_dispatch[s, t])),
                    "pv_dispatch_mw": float(pyo.value(model.pv_dispatch[s, t])),
                    "wind_curtailment_mw": float(pyo.value(model.wind_curtailment[s, t])),
                    "pv_curtailment_mw": float(pyo.value(model.pv_curtailment[s, t])),
                    "alk_power_mw": float(pyo.value(model.p_device[s, t, "ALK"])),
                    "pem_power_mw": float(pyo.value(model.p_device[s, t, "PEM"])),
                    "nh3_power_mw": float(pyo.value(model.p_device[s, t, "NH3"])),
                    "alk_on": int(round(pyo.value(model.on[t, "ALK"]))),
                    "pem_on": int(round(pyo.value(model.on[t, "PEM"]))),
                    "nh3_on": int(round(pyo.value(model.on[t, "NH3"]))),
                    "alk_startup": int(round(pyo.value(model.startup[t, "ALK"]))),
                    "pem_startup": int(round(pyo.value(model.startup[t, "PEM"]))),
                    "nh3_startup": int(round(pyo.value(model.startup[t, "NH3"]))),
                    "alk_h2_kg": float(pyo.value(model.h2_production[s, t, "ALK"])),
                    "pem_h2_kg": float(pyo.value(model.h2_production[s, t, "PEM"])),
                    "nh3_kg": float(pyo.value(model.nh3_production[s, t])),
                    "grid_buy_mw": float(pyo.value(model.grid_buy[s, t])),
                    "grid_sell_mw": float(pyo.value(model.grid_sell[s, t])),
                    "battery_charge_mw": float(pyo.value(model.charge[s, t])),
                    "battery_discharge_mw": float(pyo.value(model.discharge[s, t])),
                    "soc_mwh": float(pyo.value(model.soc[s, t])),
                    "buy_price_yuan_per_kwh": float(input_data.price_yuan_per_kwh[int(t)]),
                }
            )
    return pd.DataFrame(rows)


def validate_dispatch(dispatch: pd.DataFrame, config: dict, tolerance: float = 1e-5) -> dict:
    supply = (
        dispatch["wind_dispatch_mw"]
        + dispatch["pv_dispatch_mw"]
        + dispatch["grid_buy_mw"]
        + dispatch["battery_discharge_mw"]
    )
    demand = (
        dispatch["load_mw"]
        + dispatch["alk_power_mw"]
        + dispatch["pem_power_mw"]
        + dispatch["nh3_power_mw"]
        + dispatch["battery_charge_mw"]
        + dispatch["grid_sell_mw"]
    )
    electricity_residual = supply - demand
    hydrogen_residual = (
        dispatch["alk_h2_kg"]
        + dispatch["pem_h2_kg"]
        - float(config["equipment"]["NH3"]["hydrogen_kg_per_kg_nh3"])
        * dispatch["nh3_kg"]
    )
    simultaneous_battery = np.minimum(
        dispatch["battery_charge_mw"], dispatch["battery_discharge_mw"]
    )
    simultaneous_grid = np.minimum(dispatch["grid_buy_mw"], dispatch["grid_sell_mw"])
    target = float(config["project"]["target_nh3_tpd"])
    production_error = (
        dispatch.groupby("scenario")["nh3_kg"].sum() / 1000.0 - target
    ).abs()
    report = {
        "max_abs_electricity_balance_mw": float(electricity_residual.abs().max()),
        "max_abs_hydrogen_balance_kg": float(hydrogen_residual.abs().max()),
        "max_simultaneous_charge_discharge_mw": float(simultaneous_battery.max()),
        "max_simultaneous_grid_buy_sell_mw": float(simultaneous_grid.max()),
        "max_abs_daily_nh3_target_error_t": float(production_error.max()),
    }
    report["passed"] = bool(max(report.values()) <= tolerance)
    return report


def scenario_metrics(
    model: pyo.ConcreteModel,
    dispatch: pd.DataFrame,
    config: dict,
    plan_name: str,
) -> pd.DataFrame:
    target_tpd = float(config["project"]["target_nh3_tpd"])
    battery_daily_cost = float(pyo.value(model.annual_battery_cost)) / 360.0
    records: list[dict] = []
    for s in model.S:
        frame = dispatch[dispatch["scenario"] == str(s)]
        renewable_available = float(
            frame["wind_available_mw"].sum() + frame["pv_available_mw"].sum()
        )
        curtailment = float(
            frame["wind_curtailment_mw"].sum() + frame["pv_curtailment_mw"].sum()
        )
        energy_use = float(
            frame["load_mw"].sum()
            + frame["alk_power_mw"].sum()
            + frame["pem_power_mw"].sum()
            + frame["nh3_power_mw"].sum()
            + frame["battery_charge_mw"].sum()
        )
        grid_buy = float(frame["grid_buy_mw"].sum())
        grid_sell = float(frame["grid_sell_mw"].sum())
        rho1 = (energy_use - grid_sell - grid_buy) / renewable_available
        rho2 = (renewable_available - grid_sell) / energy_use
        rho3 = grid_sell / renewable_available
        operating_cost = float(pyo.value(model.daily_operating_cost[s]))
        records.append(
            {
                "plan": plan_name,
                "scenario": str(s),
                "daily_operating_cost_yuan": operating_cost,
                "daily_cost_with_capital_yuan": operating_cost + battery_daily_cost,
                "cost_per_ton_nh3_yuan": (operating_cost + battery_daily_cost) / target_tpd,
                "grid_import_mwh": grid_buy,
                "grid_export_mwh": grid_sell,
                "renewable_available_mwh": renewable_available,
                "curtailment_mwh": curtailment,
                "curtailment_rate": curtailment / renewable_available,
                "official_self_use_ratio": rho1,
                "official_green_power_ratio": rho2,
                "official_export_ratio": rho3,
                "all_official_metrics_pass": bool(rho1 > 0.60 and rho2 > 0.30 and rho3 < 0.20),
                "alk_h2_share": float(frame["alk_h2_kg"].sum())
                / float(frame["alk_h2_kg"].sum() + frame["pem_h2_kg"].sum()),
            }
        )
    return pd.DataFrame(records)


def planning_summary(
    deterministic_model: pyo.ConcreteModel,
    stochastic_model: pyo.ConcreteModel,
    deterministic_metrics: pd.DataFrame,
    stochastic_metrics: pd.DataFrame,
) -> pd.DataFrame:
    def summarize(name: str, model: pyo.ConcreteModel, frame: pd.DataFrame) -> dict:
        daily = frame["daily_cost_with_capital_yuan"].to_numpy()
        annual_cost = float(pyo.value(model.annual_battery_cost)) + 15.0 * float(
            frame["daily_operating_cost_yuan"].sum()
        )
        return {
            "plan": name,
            "battery_energy_mwh": float(pyo.value(model.battery_energy)),
            "battery_power_mw": float(pyo.value(model.battery_power)),
            "expected_annual_cost_yuan": annual_cost,
            "mean_daily_cost_yuan": float(np.mean(daily)),
            "p90_daily_cost_yuan": float(np.quantile(daily, 0.90)),
            "worst_daily_cost_yuan": float(np.max(daily)),
            "daily_cost_std_yuan": float(np.std(daily, ddof=0)),
            "mean_grid_import_mwh": float(frame["grid_import_mwh"].mean()),
            "mean_curtailment_rate": float(frame["curtailment_rate"].mean()),
            "official_compliance_rate": float(frame["all_official_metrics_pass"].mean()),
            "mean_alk_h2_share": float(frame["alk_h2_share"].mean()),
        }

    return pd.DataFrame(
        [
            summarize("deterministic_expected_profile", deterministic_model, deterministic_metrics),
            summarize("stochastic_24_scenarios", stochastic_model, stochastic_metrics),
        ]
    )

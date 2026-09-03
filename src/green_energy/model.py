from __future__ import annotations

from collections.abc import Mapping

import pyomo.environ as pyo

from .data import InputData, Scenario


ELECTROLYZERS = ("ALK", "PEM")
DEVICES = ("ALK", "PEM", "NH3")


def _window_ending_at(hour: int, length: int, horizon: int = 24) -> list[int]:
    return [((hour - offset) % horizon) for offset in range(length)]


def build_model(
    input_data: InputData,
    scenarios: Mapping[str, Scenario],
    config: dict,
    scenario_days: Mapping[str, float],
    fixed_battery: tuple[float, float] | None = None,
    fixed_commitment: Mapping[tuple[int, str], int] | None = None,
) -> pyo.ConcreteModel:
    """Build a device-level MILP with shared battery planning decisions.

    Battery energy/power and device commitment are first-stage variables.
    Device loading, grid exchanges and battery operation are scenario-specific
    recourse decisions.
    """
    model = pyo.ConcreteModel("green_power_hydrogen_ammonia")
    scenario_names = list(scenarios)
    model.S = pyo.Set(initialize=scenario_names, ordered=True)
    model.T = pyo.RangeSet(0, 23)
    model.E = pyo.Set(initialize=ELECTROLYZERS, ordered=True)
    model.D = pyo.Set(initialize=DEVICES, ordered=True)

    equipment = config["equipment"]
    battery = config["battery"]
    grid = config["grid"]
    renewables = config["renewables"]

    wind = {(s, t): float(scenarios[s].wind_mw[t]) for s in scenario_names for t in range(24)}
    pv = {(s, t): float(scenarios[s].pv_mw[t]) for s in scenario_names for t in range(24)}
    load = {(s, t): float(input_data.load_mw[t]) for s in scenario_names for t in range(24)}
    price = {(s, t): float(input_data.price_yuan_per_kwh[t]) for s in scenario_names for t in range(24)}

    model.wind_available = pyo.Param(model.S, model.T, initialize=wind)
    model.pv_available = pyo.Param(model.S, model.T, initialize=pv)
    model.base_load = pyo.Param(model.S, model.T, initialize=load)
    model.buy_price = pyo.Param(model.S, model.T, initialize=price)
    model.scenario_days = pyo.Param(model.S, initialize={s: float(scenario_days[s]) for s in scenario_names})

    max_energy = float(battery["max_energy_mwh"])
    max_power = float(battery["max_power_mw"])
    model.battery_energy = pyo.Var(bounds=(0.0, max_energy))
    model.battery_power = pyo.Var(bounds=(0.0, max_power))
    if fixed_battery is not None:
        model.battery_energy.fix(float(fixed_battery[0]))
        model.battery_power.fix(float(fixed_battery[1]))

    model.p_device = pyo.Var(model.S, model.T, model.D, domain=pyo.NonNegativeReals)
    model.on = pyo.Var(model.T, model.D, domain=pyo.Binary)
    model.startup = pyo.Var(model.T, model.D, domain=pyo.Binary)
    model.shutdown = pyo.Var(model.T, model.D, domain=pyo.Binary)
    if fixed_commitment is not None:
        for t in range(24):
            for device in DEVICES:
                model.on[t, device].fix(int(fixed_commitment[(t, device)]))
    model.h2_production = pyo.Var(model.S, model.T, model.E, domain=pyo.NonNegativeReals)
    model.nh3_production = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)

    model.wind_dispatch = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.pv_dispatch = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.wind_curtailment = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.pv_curtailment = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.grid_buy = pyo.Var(model.S, model.T, bounds=(0.0, float(grid["import_limit_mw"])))
    model.grid_sell = pyo.Var(model.S, model.T, bounds=(0.0, float(grid["export_limit_mw"])))
    model.grid_import_mode = pyo.Var(model.S, model.T, domain=pyo.Binary)

    model.charge = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.discharge = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.S, model.T, domain=pyo.NonNegativeReals)

    def renewable_split_wind(m, s, t):
        return m.wind_dispatch[s, t] + m.wind_curtailment[s, t] == m.wind_available[s, t]

    def renewable_split_pv(m, s, t):
        return m.pv_dispatch[s, t] + m.pv_curtailment[s, t] == m.pv_available[s, t]

    model.wind_split = pyo.Constraint(model.S, model.T, rule=renewable_split_wind)
    model.pv_split = pyo.Constraint(model.S, model.T, rule=renewable_split_pv)

    def device_lower(m, s, t, device):
        spec = equipment[device]
        return m.p_device[s, t, device] >= float(spec["rated_power_mw"]) * float(
            spec["min_load_fraction"]
        ) * m.on[t, device]

    def device_upper(m, s, t, device):
        return m.p_device[s, t, device] <= float(equipment[device]["rated_power_mw"]) * m.on[
            t, device
        ]

    model.device_lower = pyo.Constraint(model.S, model.T, model.D, rule=device_lower)
    model.device_upper = pyo.Constraint(model.S, model.T, model.D, rule=device_upper)

    def state_transition(m, t, device):
        prev = (t - 1) % 24
        return m.on[t, device] - m.on[prev, device] == m.startup[t, device] - m.shutdown[t, device]

    model.state_transition = pyo.Constraint(model.T, model.D, rule=state_transition)
    model.no_simultaneous_start_stop = pyo.Constraint(
        model.T,
        model.D,
        rule=lambda m, t, d: m.startup[t, d] + m.shutdown[t, d] <= 1,
    )

    def ramp_up(m, s, t, device):
        prev = (t - 1) % 24
        spec = equipment[device]
        rated = float(spec["rated_power_mw"])
        ramp = rated * float(spec["ramp_up_fraction_per_hour"])
        return m.p_device[s, t, device] - m.p_device[s, prev, device] <= ramp * m.on[
            prev, device
        ] + rated * m.startup[t, device]

    def ramp_down(m, s, t, device):
        prev = (t - 1) % 24
        spec = equipment[device]
        rated = float(spec["rated_power_mw"])
        ramp = rated * float(spec["ramp_down_fraction_per_hour"])
        return m.p_device[s, prev, device] - m.p_device[s, t, device] <= ramp * m.on[
            t, device
        ] + rated * m.shutdown[t, device]

    model.ramp_up = pyo.Constraint(model.S, model.T, model.D, rule=ramp_up)
    model.ramp_down = pyo.Constraint(model.S, model.T, model.D, rule=ramp_down)

    def minimum_up(m, t, device):
        length = int(equipment[device]["min_up_hours"])
        return sum(m.startup[tau, device] for tau in _window_ending_at(t, length)) <= m.on[t, device]

    def minimum_down(m, t, device):
        length = int(equipment[device]["min_down_hours"])
        return sum(m.shutdown[tau, device] for tau in _window_ending_at(t, length)) <= 1 - m.on[t, device]

    model.minimum_up = pyo.Constraint(model.T, model.D, rule=minimum_up)
    model.minimum_down = pyo.Constraint(model.T, model.D, rule=minimum_down)

    def hydrogen_conversion(m, s, t, electrolyzer):
        spec = equipment[electrolyzer]
        kg_per_mwh = 1000.0 * float(spec["efficiency"]) / float(
            spec["base_specific_energy_kwh_per_kg_h2"]
        )
        return m.h2_production[s, t, electrolyzer] == kg_per_mwh * m.p_device[
            s, t, electrolyzer
        ]

    model.hydrogen_conversion = pyo.Constraint(model.S, model.T, model.E, rule=hydrogen_conversion)

    nh3_spec = equipment["NH3"]
    electricity_per_kg_nh3 = float(nh3_spec["electricity_kwh_per_kg_nh3"])
    hydrogen_per_kg_nh3 = float(nh3_spec["hydrogen_kg_per_kg_nh3"])
    model.nh3_electricity = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: m.p_device[s, t, "NH3"]
        == electricity_per_kg_nh3 * m.nh3_production[s, t] / 1000.0,
    )
    model.hydrogen_balance = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: sum(m.h2_production[s, t, e] for e in m.E)
        == hydrogen_per_kg_nh3 * m.nh3_production[s, t],
    )

    target_tpd = float(config["project"]["target_nh3_tpd"])
    model.daily_nh3_target = pyo.Constraint(
        model.S,
        rule=lambda m, s: sum(m.nh3_production[s, t] for t in m.T) == target_tpd * 1000.0,
    )

    model.power_balance = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: m.wind_dispatch[s, t]
        + m.pv_dispatch[s, t]
        + m.grid_buy[s, t]
        + m.discharge[s, t]
        == m.base_load[s, t]
        + sum(m.p_device[s, t, d] for d in m.D)
        + m.charge[s, t]
        + m.grid_sell[s, t],
    )
    model.grid_buy_mode_limit = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: m.grid_buy[s, t]
        <= float(grid["import_limit_mw"]) * m.grid_import_mode[s, t],
    )
    model.grid_sell_mode_limit = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: m.grid_sell[s, t]
        <= float(grid["export_limit_mw"]) * (1 - m.grid_import_mode[s, t]),
    )

    eta_ch = float(battery["charge_efficiency"])
    eta_dis = float(battery["discharge_efficiency"])
    self_loss = float(battery["self_loss_fraction_per_hour"])
    model.soc_transition = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: m.soc[s, t]
        == (1.0 - self_loss) * m.soc[s, (t - 1) % 24]
        + eta_ch * m.charge[s, t]
        - m.discharge[s, t] / eta_dis,
    )
    model.soc_lower = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: m.soc[s, t]
        >= float(battery["min_soc_fraction"]) * m.battery_energy,
    )
    model.soc_upper = pyo.Constraint(
        model.S,
        model.T,
        rule=lambda m, s, t: m.soc[s, t]
        <= float(battery["max_soc_fraction"]) * m.battery_energy,
    )
    model.charge_capacity = pyo.Constraint(
        model.S, model.T, rule=lambda m, s, t: m.charge[s, t] <= m.battery_power
    )
    model.discharge_capacity = pyo.Constraint(
        model.S, model.T, rule=lambda m, s, t: m.discharge[s, t] <= m.battery_power
    )
    model.minimum_duration = pyo.Constraint(
        expr=model.battery_energy >= float(battery["min_duration_hours"]) * model.battery_power
    )

    wind_cost = float(renewables["wind_cost_yuan_per_kwh"])
    pv_cost = float(renewables["pv_cost_yuan_per_kwh"])
    sell_price = float(grid["sell_price_yuan_per_kwh"])
    throughput_cost = float(battery["om_yuan_per_kwh_throughput"])

    def daily_operating_cost(m, s):
        energy_market = sum(
            1000.0
            * (
                m.buy_price[s, t] * m.grid_buy[s, t]
                - sell_price * m.grid_sell[s, t]
                + wind_cost * m.wind_dispatch[s, t]
                + pv_cost * m.pv_dispatch[s, t]
            )
            for t in m.T
        )
        device_om = sum(
            1000.0 * float(equipment[d]["om_yuan_per_kwh"]) * m.p_device[s, t, d]
            for t in m.T
            for d in m.D
        )
        startup_cost = sum(
            float(equipment[d]["startup_cost_yuan"]) * m.startup[t, d]
            for t in m.T
            for d in m.D
        )
        battery_om = sum(
            1000.0 * throughput_cost * (m.charge[s, t] + m.discharge[s, t]) for t in m.T
        )
        return energy_market + device_om + startup_cost + battery_om

    model.daily_operating_cost = pyo.Expression(model.S, rule=daily_operating_cost)
    annualized_battery_cost_per_mwh = (
        float(battery["energy_cost_yuan_per_kwh"]) * 1000.0 / float(battery["lifetime_years"])
    )
    model.annual_battery_cost = pyo.Expression(
        expr=annualized_battery_cost_per_mwh * model.battery_energy
    )
    model.total_annual_cost = pyo.Expression(
        expr=model.annual_battery_cost
        + sum(model.scenario_days[s] * model.daily_operating_cost[s] for s in model.S)
    )
    model.objective = pyo.Objective(expr=model.total_annual_cost, sense=pyo.minimize)
    return model

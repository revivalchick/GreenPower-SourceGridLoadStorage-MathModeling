from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyomo.environ as pyo

from .analysis import (
    extract_dispatch,
    planning_summary,
    scenario_metrics,
    validate_dispatch,
)
from .data import load_config, load_input_data, mean_scenario
from .model import build_model
from .plotting import (
    plot_battery,
    plot_cost_risk,
    plot_device_dispatch,
    plot_energy_balance,
)
from .solve import solve_model


def run_pipeline(project_root: str | Path) -> dict:
    project_root = Path(project_root)
    config = load_config(project_root / "configs" / "base.yaml")
    data = load_input_data(project_root / "data" / "processed", config)
    output_dir = project_root / "outputs"
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    typical_model = build_model(
        data,
        {data.typical.name: data.typical},
        config,
        {data.typical.name: 1.0},
        fixed_battery=(0.0, 0.0),
    )
    typical_solve = solve_model(typical_model, config)
    typical_dispatch = extract_dispatch(typical_model, data)
    typical_validation = validate_dispatch(typical_dispatch, config)
    if not typical_validation["passed"]:
        raise RuntimeError(f"Typical-day validation failed: {typical_validation}")
    typical_dispatch.to_csv(output_dir / "typical_device_level_dispatch.csv", index=False)

    expected = mean_scenario(data.scenarios)
    deterministic_plan_model = build_model(
        data,
        {expected.name: expected},
        config,
        {expected.name: 360.0},
    )
    deterministic_plan_solve = solve_model(deterministic_plan_model, config)
    deterministic_capacity = (
        deterministic_plan_solve.battery_energy_mwh,
        deterministic_plan_solve.battery_power_mw,
    )
    deterministic_commitment = {
        (t, device): int(round(pyo.value(deterministic_plan_model.on[t, device])))
        for t in deterministic_plan_model.T
        for device in deterministic_plan_model.D
    }

    scenario_days = {name: float(config["project"]["days_per_scenario"]) for name in data.scenarios}
    deterministic_eval_model = build_model(
        data,
        data.scenarios,
        config,
        scenario_days,
        fixed_battery=deterministic_capacity,
        fixed_commitment=deterministic_commitment,
    )
    deterministic_eval_solve = solve_model(deterministic_eval_model, config)
    deterministic_dispatch = extract_dispatch(deterministic_eval_model, data)
    deterministic_validation = validate_dispatch(deterministic_dispatch, config)
    if not deterministic_validation["passed"]:
        raise RuntimeError(f"Deterministic evaluation validation failed: {deterministic_validation}")
    deterministic_metrics = scenario_metrics(
        deterministic_eval_model, deterministic_dispatch, config, "deterministic_expected_profile"
    )

    stochastic_model = build_model(data, data.scenarios, config, scenario_days)
    stochastic_solve = solve_model(stochastic_model, config)
    stochastic_dispatch = extract_dispatch(stochastic_model, data)
    stochastic_validation = validate_dispatch(stochastic_dispatch, config)
    if not stochastic_validation["passed"]:
        raise RuntimeError(f"Stochastic validation failed: {stochastic_validation}")
    stochastic_metrics = scenario_metrics(
        stochastic_model, stochastic_dispatch, config, "stochastic_24_scenarios"
    )

    comparison = planning_summary(
        deterministic_eval_model,
        stochastic_model,
        deterministic_metrics,
        stochastic_metrics,
    )
    deterministic_metrics.to_csv(output_dir / "deterministic_scenario_metrics.csv", index=False)
    stochastic_metrics.to_csv(output_dir / "stochastic_scenario_metrics.csv", index=False)
    deterministic_dispatch.to_csv(output_dir / "deterministic_scenario_dispatch.csv", index=False)
    stochastic_dispatch.to_csv(output_dir / "stochastic_scenario_dispatch.csv", index=False)
    comparison.to_csv(output_dir / "planning_comparison.csv", index=False)

    representative = stochastic_metrics.sort_values("daily_cost_with_capital_yuan").iloc[
        len(stochastic_metrics) // 2
    ]["scenario"]
    plot_device_dispatch(typical_dispatch, figure_dir, "typical")
    plot_energy_balance(stochastic_dispatch, figure_dir, str(representative))
    plot_battery(stochastic_dispatch, figure_dir, str(representative))
    plot_cost_risk(deterministic_metrics, stochastic_metrics, figure_dir)

    vss = float(
        comparison.loc[
            comparison["plan"] == "deterministic_expected_profile", "expected_annual_cost_yuan"
        ].iloc[0]
        - comparison.loc[
            comparison["plan"] == "stochastic_24_scenarios", "expected_annual_cost_yuan"
        ].iloc[0]
    )
    run_summary = {
        "typical_solve": typical_solve.__dict__,
        "deterministic_plan_solve": deterministic_plan_solve.__dict__,
        "deterministic_evaluation_solve": deterministic_eval_solve.__dict__,
        "stochastic_solve": stochastic_solve.__dict__,
        "value_of_stochastic_solution_yuan_per_year": vss,
        "representative_stochastic_scenario": str(representative),
        "validation": {
            "typical": typical_validation,
            "deterministic_evaluation": deterministic_validation,
            "stochastic": stochastic_validation,
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return run_summary

from __future__ import annotations

from dataclasses import dataclass

import pyomo.environ as pyo


@dataclass(frozen=True)
class SolveSummary:
    status: str
    termination_condition: str
    objective_yuan_per_year: float
    battery_energy_mwh: float
    battery_power_mw: float


def solve_model(model: pyo.ConcreteModel, config: dict) -> SolveSummary:
    solver_name = config["solver"].get("name", "highs")
    solver = pyo.SolverFactory(solver_name)
    if not solver.available(exception_flag=False):
        raise RuntimeError(f"Solver '{solver_name}' is not available")
    solver.options["mip_rel_gap"] = float(config["solver"].get("mip_rel_gap", 0.005))
    solver.options["time_limit"] = float(config["solver"].get("time_limit_seconds", 240))
    result = solver.solve(model, tee=False)
    termination = str(result.solver.termination_condition)
    if termination.lower() not in {"optimal", "feasible", "maxTimeLimit".lower()}:
        raise RuntimeError(f"MILP solve failed: {result.solver.status}, {termination}")
    return SolveSummary(
        status=str(result.solver.status),
        termination_condition=termination,
        objective_yuan_per_year=float(pyo.value(model.total_annual_cost)),
        battery_energy_mwh=float(pyo.value(model.battery_energy)),
        battery_power_mw=float(pyo.value(model.battery_power)),
    )


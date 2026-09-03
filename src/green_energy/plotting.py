from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _prepare(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return path


def plot_device_dispatch(dispatch: pd.DataFrame, output_dir: str | Path, scenario: str) -> Path:
    output_dir = _prepare(output_dir)
    frame = dispatch[dispatch["scenario"] == scenario].sort_values("hour")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(frame["hour"], frame["alk_power_mw"], marker="o", label="ALK功率")
    axes[0].plot(frame["hour"], frame["pem_power_mw"], marker="o", label="PEM功率")
    axes[0].plot(frame["hour"], frame["nh3_power_mw"], marker="o", label="合成氨功率")
    axes[0].set_ylabel("功率 / MW")
    axes[0].set_title(f"设备级MILP调度：{scenario}")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3)
    axes[1].step(frame["hour"], frame["alk_on"], where="mid", label="ALK状态")
    axes[1].step(frame["hour"], frame["pem_on"] + 1.2, where="mid", label="PEM状态（平移）")
    axes[1].step(frame["hour"], frame["nh3_on"] + 2.4, where="mid", label="合成氨状态（平移）")
    axes[1].set_xlabel("时段 / h")
    axes[1].set_ylabel("启停状态")
    axes[1].set_xticks(np.arange(0, 24, 1))
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3)
    path = output_dir / "device_level_dispatch.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_energy_balance(dispatch: pd.DataFrame, output_dir: str | Path, scenario: str) -> Path:
    output_dir = _prepare(output_dir)
    frame = dispatch[dispatch["scenario"] == scenario].sort_values("hour")
    x = frame["hour"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.stackplot(
        x,
        frame["wind_dispatch_mw"],
        frame["pv_dispatch_mw"],
        frame["grid_buy_mw"],
        frame["battery_discharge_mw"],
        labels=["风电", "光伏", "电网购电", "储能放电"],
        alpha=0.82,
    )
    demand = (
        frame["load_mw"]
        + frame["alk_power_mw"]
        + frame["pem_power_mw"]
        + frame["nh3_power_mw"]
        + frame["battery_charge_mw"]
        + frame["grid_sell_mw"]
    )
    ax.plot(x, demand, color="black", linewidth=2.0, label="总需求（含充电/售电）")
    ax.set_title(f"24小时电力供需平衡：{scenario}")
    ax.set_xlabel("时段 / h")
    ax.set_ylabel("功率 / MW")
    ax.set_xticks(np.arange(0, 24, 1))
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    path = output_dir / "power_balance.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_battery(dispatch: pd.DataFrame, output_dir: str | Path, scenario: str) -> Path:
    output_dir = _prepare(output_dir)
    frame = dispatch[dispatch["scenario"] == scenario].sort_values("hour")
    fig, ax1 = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax1.bar(frame["hour"], frame["battery_charge_mw"], width=0.7, label="充电", color="#2A9D8F")
    ax1.bar(
        frame["hour"],
        -frame["battery_discharge_mw"],
        width=0.7,
        label="放电",
        color="#E76F51",
    )
    ax1.set_xlabel("时段 / h")
    ax1.set_ylabel("充放电功率 / MW")
    ax1.grid(alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(frame["hour"], frame["soc_mwh"], color="#264653", marker="o", label="SOC")
    ax2.set_ylabel("SOC / MWh")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncol=3, loc="upper right")
    ax1.set_title(f"储能充放电与SOC：{scenario}")
    path = output_dir / "battery_soc.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cost_risk(
    deterministic_metrics: pd.DataFrame,
    stochastic_metrics: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    output_dir = _prepare(output_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    data = [
        deterministic_metrics["daily_cost_with_capital_yuan"].to_numpy(),
        stochastic_metrics["daily_cost_with_capital_yuan"].to_numpy(),
    ]
    box = ax.boxplot(
        data,
        tick_labels=["确定性容量方案", "24场景随机规划"],
        patch_artist=True,
    )
    for patch, color in zip(box["boxes"], ["#F4A261", "#2A9D8F"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("场景日成本（含日均资本成本）/ 元")
    ax.set_title("确定性与随机规划的场景成本风险")
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "deterministic_vs_stochastic_cost_risk.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

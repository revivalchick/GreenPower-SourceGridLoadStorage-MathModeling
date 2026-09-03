"""Convert the eight competition spreadsheets into small, auditable CSV files.

The raw Excel files are intentionally not copied into this repository. Adjust
SOURCE_DIR when reproducing the conversion on another computer.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(r"D:\数模竞赛\A题")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def read_second_column(filename: str, target_name: str) -> pd.DataFrame:
    frame = pd.read_excel(SOURCE_DIR / filename)
    return pd.DataFrame({"hour": range(24), target_name: frame.iloc[:, 1].astype(float)})


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    read_second_column(
        "附件1：园区典型日常规电负荷标幺功率曲线.xlsx", "load_pu"
    ).to_csv(OUTPUT_DIR / "load.csv", index=False)

    typical = pd.read_excel(SOURCE_DIR / "附件2：典型日风电、光伏标幺功率表.xlsx")
    pd.DataFrame(
        {
            "hour": range(24),
            "wind_pu": typical.iloc[:, 1].astype(float),
            "pv_pu": typical.iloc[:, 2].astype(float),
        }
    ).to_csv(OUTPUT_DIR / "typical.csv", index=False)

    wind = pd.read_excel(SOURCE_DIR / "附件3：园区6种场景的风电标幺功率表.xlsx")
    wind_out = pd.DataFrame({"hour": range(24)})
    for index in range(6):
        wind_out[f"wind_{index + 1}"] = wind.iloc[:, index + 1].astype(float)
    wind_out.to_csv(OUTPUT_DIR / "wind_scenarios.csv", index=False)

    pv = pd.read_excel(SOURCE_DIR / "附件4：园区4种场景的光伏标幺功率表.xlsx")
    pv_out = pd.DataFrame({"hour": range(24)})
    for index in range(4):
        pv_out[f"pv_{index + 1}"] = pv.iloc[:, index + 1].astype(float)
    pv_out.to_csv(OUTPUT_DIR / "pv_scenarios.csv", index=False)

    prices = []
    for hour in range(24):
        if 10 <= hour < 15 or 18 <= hour < 21:
            prices.append(0.8024)
        elif 7 <= hour < 10 or 15 <= hour < 18 or 21 <= hour < 23:
            prices.append(0.6074)
        else:
            prices.append(0.3424)
    pd.DataFrame({"hour": range(24), "buy_price_yuan_per_kwh": prices}).to_csv(
        OUTPUT_DIR / "prices.csv", index=False
    )

    print(f"Processed data written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


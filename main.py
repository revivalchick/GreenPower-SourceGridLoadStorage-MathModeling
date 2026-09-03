from pathlib import Path

from green_energy.pipeline import run_pipeline


if __name__ == "__main__":
    summary = run_pipeline(Path(__file__).resolve().parent)
    print("Optimization completed.")
    print(f"VSS: {summary['value_of_stochastic_solution_yuan_per_year']:.2f} yuan/year")


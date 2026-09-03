# Data notes

The committed CSV files are processed from the eight spreadsheets supplied with
the 2026 competition problem "绿电直连型电氢氨园区优化运行".

- `load.csv`: 24-hour normalized conventional load.
- `typical.csv`: typical wind and photovoltaic profiles.
- `wind_scenarios.csv`: six wind scenarios.
- `pv_scenarios.csv`: four photovoltaic scenarios.
- `prices.csv`: hourly time-of-use purchase price.

The six wind and four photovoltaic profiles form 24 combinations. The problem
statement assigns 15 days to every combination, so the stochastic model uses
equal scenario weights.

The original spreadsheets are not redistributed in this repository. The
conversion script records the expected source filenames for reproducibility.


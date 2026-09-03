# Model formulation

## Indices

- `t = 0,...,23`: hourly period.
- `s = 1,...,24`: wind-photovoltaic scenario.
- `i in {ALK, PEM, NH3}`: equipment.

## First-stage decisions

- Battery energy capacity `E_B` and power capacity `P_B`.
- Common day-ahead commitment, startup and shutdown states for ALK, PEM and NH3 equipment.

## Second-stage decisions

- Equipment loading, renewable dispatch and curtailment.
- Grid import/export.
- Battery charge/discharge and state of charge.
- Hydrogen and ammonia production.

## Key constraints

### Equipment operating region

```text
P_min[i] * u[t,i] <= P[s,t,i] <= P_max[i] * u[t,i]
u[t,i] - u[t-1,i] = startup[t,i] - shutdown[t,i]
```

Minimum up/down time uses cyclic rolling windows over the representative day.
Ramp constraints couple every scenario's equipment loading across adjacent hours.

### Hydrogen-ammonia coupling

```text
H[s,t,ALK] = 1000 * eta_ALK * P[s,t,ALK] / 50
H[s,t,PEM] = 1000 * eta_PEM * P[s,t,PEM] / 50
H[s,t,ALK] + H[s,t,PEM] = 0.2 * Q_NH3[s,t]
P[s,t,NH3] = 0.5 * Q_NH3[s,t] / 1000
sum_t Q_NH3[s,t] = 54,000 kg
```

### Electricity balance

```text
wind + pv + grid_buy + battery_discharge
= conventional_load + ALK + PEM + NH3 + battery_charge + grid_sell
```

Grid import and export are mutually exclusive. This constraint is essential because the low-valley purchase tariff is below the export tariff.

### Battery

```text
SOC[s,t] = (1-loss)SOC[s,t-1] + eta_ch*charge[s,t] - discharge[s,t]/eta_dis
SOC_min*E_B <= SOC[s,t] <= SOC_max*E_B
charge[s,t], discharge[s,t] <= P_B
E_B >= 2*P_B
```

The SOC equation is cyclic over 24 hours.

## Objective

```text
min annualized_battery_cost
  + sum_s days[s] * daily_operating_cost[s]
```

Daily operating cost includes grid purchases, renewable generation costs,
equipment O&M, startup costs, battery throughput O&M and export revenue.

The stochastic model minimizes expected annual cost. P90, worst-case and
standard deviation are reported as risk diagnostics; they are not additional
optimization objectives.

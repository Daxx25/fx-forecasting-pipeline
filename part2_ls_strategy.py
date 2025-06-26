
"""
Part 2 - Long/Short Strategy Based on Slope of Price Trend
Author: Daksh Parekh
Date: May 1, 2025

Strategy:
1. Use simulated price data from Part 1.
2. For each hour at H5, H6, H7, run 20-point linear regression on USDJPY (adjusted) and GBPUSD.
3. Go long on the one with higher slope, short on the lower.
4. Close positions at hour H8, and compute profit/loss assuming $100 per step.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# -----------------------
# STEP 1: Load Simulated Prices
# -----------------------
np.random.seed(42)
start_time = datetime.datetime(2025, 5, 1, 9, 0)
timestamps = [start_time + datetime.timedelta(minutes=i*6) for i in range(60)]  # 6-min intervals = 6/hr * 10 hrs

price_data = pd.DataFrame({
    "timestamp": timestamps,
    "USDJPY": np.round(np.random.normal(143.028, 0.3, 60), 3),
    "GBPUSD": np.round(np.random.normal(1.33246, 0.005, 60), 5)
})

# Normalize USDJPY by dividing by 100
price_data["USDJPY_adj"] = price_data["USDJPY"] / 100

# -----------------------
# STEP 2: Run Slope-Based L/S Strategy
# -----------------------
results = []
hours = [5, 6, 7]
entry_indices = [h*10 for h in hours]  # 10 data points per hour
exit_index = 59  # last index for hour 8

for i, idx in enumerate(entry_indices):
    if idx + 20 > len(price_data):
        break
    subset = price_data.iloc[idx-20:idx]  # 20-point regression window
    
    # Regression for GBPUSD
    X = np.arange(20).reshape(-1, 1)
    y_gbp = subset["GBPUSD"].values.reshape(-1, 1)
    gbp_model = LinearRegression().fit(X, y_gbp)
    slope_gbp = gbp_model.coef_[0][0]

    # Regression for USDJPY (adjusted)
    y_jpy = subset["USDJPY_adj"].values.reshape(-1, 1)
    jpy_model = LinearRegression().fit(X, y_jpy)
    slope_jpy = jpy_model.coef_[0][0]

    long_cp = "GBPUSD" if slope_gbp > slope_jpy else "USDJPY"
    short_cp = "USDJPY" if long_cp == "GBPUSD" else "GBPUSD"

    entry_price_long = price_data.iloc[idx][long_cp] / (100 if long_cp == "USDJPY" else 1)
    entry_price_short = price_data.iloc[idx][short_cp] / (100 if short_cp == "USDJPY" else 1)
    exit_price_long = price_data.iloc[exit_index][long_cp] / (100 if long_cp == "USDJPY" else 1)
    exit_price_short = price_data.iloc[exit_index][short_cp] / (100 if short_cp == "USDJPY" else 1)

    # P/L Calculation (assume $100 per trade step)
    position_value = 100
    pl_long = position_value * ((exit_price_long - entry_price_long) / entry_price_long)
    pl_short = position_value * ((entry_price_short - exit_price_short) / entry_price_short)
    total_pl = pl_long + pl_short

    results.append({
        "step": f"H{hours[i]}-H8",
        "long": long_cp,
        "short": short_cp,
        "entry_price_long": entry_price_long,
        "exit_price_long": exit_price_long,
        "entry_price_short": entry_price_short,
        "exit_price_short": exit_price_short,
        "PL_long": pl_long,
        "PL_short": pl_short,
        "Total_PL": total_pl
    })

# -----------------------
# STEP 3: Export to Excel
# -----------------------
result_df = pd.DataFrame(results)
price_data.to_excel("/mnt/data/mock_price_data.xlsx", index=False)
result_df.to_excel("/mnt/data/ls_strategy_results.xlsx", index=False)

print("✅ L/S Strategy simulation complete. Results exported to Excel.")

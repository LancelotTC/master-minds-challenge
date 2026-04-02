import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### CONFIG

FILE_PATH = "predictions/XGBRegressor/XGBRegressor_validation_preds.csv"

LANE_CAPACITY = 200          # pax/hour/lane
AGENTS_PER_LANE = 8
SHIFT_HOURS = 6              # minimum duration
SMOOTHING_WINDOW = 3      # smoothing to avoid spikes
SAFETY_FACTOR = 1          # extra margin
MAX_LANES = 15            # avoid unrealistic values


### LOAD DATA

df = pd.read_csv(FILE_PATH)

# Ensure datetime is correct
df["LTScheduledDatetime"] = pd.to_datetime(df["LTScheduledDatetime"])

# Rename prediction column if needed
if "NbPaxTotalPrediction" in df.columns:
    df["predicted_pax"] = df["NbPaxTotalPrediction"]
elif "Predicted NbPaxTotal" in df.columns:
    df["predicted_pax"] = df["Predicted NbPaxTotal"]
else:
    raise ValueError("Prediction column not found")

# Real values (optional)
if "NbPaxTotal" in df.columns:
    df["real_pax"] = df["NbPaxTotal"]
else:
    df["real_pax"] = np.nan


### AGGREGATE HOURLY

df["hour"] = df["LTScheduledDatetime"].dt.floor("H")

hourly = df.groupby("hour", as_index=False).agg({
    "predicted_pax": "sum",
    "real_pax": "sum"
}).sort_values("hour")

# Fill missing hours
all_hours = pd.date_range(hourly["hour"].min(), hourly["hour"].max(), freq="H")
hourly = hourly.set_index("hour").reindex(all_hours, fill_value=0).reset_index()
hourly.rename(columns={"index": "hour"}, inplace=True)


### SMOOTH DEMAND

# Smooth to avoid opening too many lanes for spikes
hourly["smoothed_pax"] = (
    hourly["predicted_pax"]
    .rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1)
    .mean()
)

# Take max between raw and smoothed (robust)
demand = hourly["smoothed_pax"]

# Add safety margin
demand = demand * SAFETY_FACTOR


### TARGET LANES

hourly["target_lanes"] = np.ceil(demand / LANE_CAPACITY).astype(int)
hourly["target_lanes"] = hourly["target_lanes"].clip(0, MAX_LANES)


### SHIFT LOGIC (6 HOURS MIN)

shift_starts = np.zeros(len(hourly), dtype=int)

def compute_active_lanes(starts):
    """Compute active lanes based on shift starts"""
    active = np.zeros(len(starts), dtype=int)
    for t in range(len(starts)):
        start_idx = max(0, t - SHIFT_HOURS + 1)
        active[t] = starts[start_idx:t+1].sum()
    return active

for t in range(len(hourly)):
    active = compute_active_lanes(shift_starts)
    needed = hourly.loc[t, "target_lanes"]

    if needed > active[t]:
        shift_starts[t] += (needed - active[t])

# Final active lanes
hourly["shift_starts"] = shift_starts
hourly["active_lanes"] = compute_active_lanes(shift_starts)

# Capacity
hourly["capacity"] = hourly["active_lanes"] * LANE_CAPACITY


### BACKLOG (WAITING PASSENGERS)

backlog = []
current = 0

for i in range(len(hourly)):
    demand = hourly.loc[i, "predicted_pax"]
    capacity = hourly.loc[i, "capacity"]

    current = max(0, current + demand - capacity)
    backlog.append(current)

hourly["backlog"] = backlog


### GRAPHS

# GRAPH (FULL PERIOD)
plt.figure(figsize=(14,5))
plt.plot(hourly["hour"], hourly["predicted_pax"], label="Predicted")
plt.plot(hourly["hour"], hourly["real_pax"], label="Real")
plt.plot(hourly["hour"], hourly["capacity"], label="Capacity")
plt.legend()
plt.title("Demand vs Capacity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# GRAPH (LAST DAY ONLY)
last_day = hourly["hour"].dt.floor("D").max()
last_df = hourly[hourly["hour"].dt.floor("D") == last_day]

plt.figure(figsize=(14,5))
plt.plot(last_df["hour"], last_df["predicted_pax"], label="Predicted")
plt.plot(last_df["hour"], last_df["real_pax"], label="Real")
plt.plot(last_df["hour"], last_df["capacity"], label="Capacity")
plt.legend()
plt.title("Last Day Only")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# GRAPH BACKLOG
plt.figure(figsize=(14,5))
plt.plot(hourly["hour"], hourly["backlog"], label="Backlog")
plt.legend()
plt.title("Backlog (waiting passengers)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# GRAPH (LAST DAY ONLY) BACKLOG
last_day = hourly["hour"].dt.floor("D").max()
last_df = hourly[hourly["hour"].dt.floor("D") == last_day]

plt.figure(figsize=(14,5))
plt.plot(last_df["hour"],last_df["backlog"], label="Backlog")
plt.legend()
plt.title("Backlog (waiting passengers) only last day")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


### GRAPH - ACTIVE LANES ONLY FOR THE LAST DAY

last_day = hourly["hour"].dt.floor("D").max()
last_day_df = hourly[hourly["hour"].dt.floor("D") == last_day].copy()

plt.figure(figsize=(14,5))
plt.plot(last_day_df["hour"].dt.strftime("%H:%M"), last_day_df["active_lanes"])
plt.title("Active lanes by hour - last day")
plt.xlabel("Hour")
plt.ylabel("Active lanes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# GRAPH (LAST DAY ONLY) DEMAND - CAPACITY
last_day = hourly["hour"].dt.floor("D").max()
last_df = hourly[hourly["hour"].dt.floor("D") == last_day].copy()

last_df["gap"] = last_df["predicted_pax"] - last_df["capacity"]

plt.figure(figsize=(14,5))
plt.axhline(0, linestyle="--")
plt.plot(last_df["hour"], last_df["gap"], label="Predicted demand - capacity")
plt.legend()
plt.title("Last day: demand minus capacity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
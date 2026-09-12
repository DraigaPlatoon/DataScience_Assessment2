import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Load data
df = pd.read_csv(
    "world_cup_2026_player_stats.csv",
    header=[0, 1]
)

df.columns = df.columns.get_level_values(1)

# Rename the second G+A column to G+A_per90
cols = list(df.columns)
ga_positions = [i for i, col in enumerate(cols) if col == "G+A"]

cols[ga_positions[1]] = "G+A_per90"
df.columns = cols

# Convert columns to numbers
for col in ["MP", "Starts", "Min", "G+A_per90"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep players with at least 90 minutes
df = df[df["Min"] >= 90].copy()

# Create start rate
df["Start_Rate"] = df["Starts"] / df["MP"]

# Create groups
starters = df[df["Start_Rate"] >= 0.50]
substitutes = df[df["Start_Rate"] < 0.50]

# Take same random samples
starter_sample = starters.sample(n=60, random_state=42)
substitute_sample = substitutes.sample(n=60, random_state=42)

# Create boxplot
plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        starter_sample["G+A_per90"],
        substitute_sample["G+A_per90"]
    ],
    tick_labels=[
        "Regular Starters",
        "Primarily Substitutes"
    ]
)

plt.ylabel("Goals + Assists per 90 minutes")
plt.title("G+A per 90 by Playing Role")

plt.tight_layout()

plt.savefig(
    "ga_per90_boxplot.png",
    dpi=300
)

plt.close()

print("SUCCESS - boxplot created")
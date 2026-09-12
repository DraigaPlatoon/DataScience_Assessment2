import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("./datasheet.csv")
uefa = df[df["Confederation"] == "UEFA"]["Shots_per_90"]
non_uefa = df[df["Confederation"] == "Non-UEFA"]["Shots_per_90"]

# Plot dual histogram
plt.figure(figsize=(7, 5))
plt.hist(
    uefa,
    bins=6,
    alpha=0.65,
    color="#4A90E2",
    label=f"UEFA (n={len(uefa)})",
    edgecolor="black",
)
plt.hist(
    non_uefa,
    bins=8,
    alpha=0.65,
    color="#F5A623",
    label=f"Non-UEFA (n={len(non_uefa)})",
    edgecolor="black",
)

# Overlay mean lines
plt.axvline(
    uefa.mean(),
    color="blue",
    linestyle="dashed",
    linewidth=2,
    label=f"UEFA Mean ({uefa.mean():.2f})",
)
plt.axvline(
    non_uefa.mean(),
    color="darkorange",
    linestyle="dashed",
    linewidth=2,
    label=f"Non-UEFA Mean ({non_uefa.mean():.2f})",
)

plt.title("Week 2: Shot Volume Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Shots per 90 Minutes")
plt.ylabel("Number of Teams")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("chart1_histogram.png", dpi=300)
plt.show()

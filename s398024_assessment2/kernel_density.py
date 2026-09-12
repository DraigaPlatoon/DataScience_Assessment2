import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("merged_fifa2026_squad_data.csv")
ko = df[df["Stage"] == "Knockout Stage"]["G/Sh"].dropna().values
elim = df[df["Stage"] == "Group Stage Eliminated"]["G/Sh"].dropna().values

plt.figure(figsize=(7, 5.5), dpi=300)

# Density curves
sns.kdeplot(
    ko,
    color="#2b5c8f",
    fill=True,
    alpha=0.35,
    linewidth=2,
    label=f"Knockout Stage (n={len(ko)})",
)
sns.kdeplot(
    elim,
    color="#d95f02",
    fill=True,
    alpha=0.35,
    linewidth=2,
    label=f"Group Eliminated (n={len(elim)})",
)

# Reference mean lines
plt.axvline(
    np.mean(ko),
    color="#2b5c8f",
    linestyle="--",
    linewidth=1.6,
    label=f"Knockout Mean: {np.mean(ko):.3f}",
)
plt.axvline(
    np.mean(elim),
    color="#d95f02",
    linestyle="--",
    linewidth=1.6,
    label=f"Eliminated Mean: {np.mean(elim):.3f}",
)

plt.title(
    "Goal Conversion Density & Distribution Shift",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
plt.xlabel(
    "Goal Conversion Efficiency (Gls / Sh)",
    fontsize=10,
    fontweight="semibold",
)
plt.ylabel(
    "Estimated Probability Density", fontsize=10, fontweight="semibold"
)
plt.legend(loc="upper right", fontsize=9, frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("chart2_density_kde.png", dpi=300)
plt.close()
print("--> Saved 'chart2_density_kde.png'!")
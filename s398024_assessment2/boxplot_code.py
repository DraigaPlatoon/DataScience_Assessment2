import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("merged_fifa2026_squad_data.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
palette = {"Knockout Stage": "#1f77b4", "Group Stage Eliminated": "#ff7f0e"}

# Panel A: Boxplot with Individual Data Points & Mean Markers
sns.boxplot(
    ax=axes[0],
    x="Stage",
    y="G/Sh",
    data=df,
    palette=palette,
    width=0.45,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "black",
        "markersize": "8",
    },
)
sns.stripplot(
    ax=axes[0],
    x="Stage",
    y="G/Sh",
    data=df,
    color="black",
    alpha=0.6,
    jitter=0.2,
    size=6,
)
axes[0].set_title(
    "A. Efficiency by Qualification Stage (Boxplot & Data Points)",
    fontsize=11,
    fontweight="bold",
)
axes[0].set_ylabel(
    "Goal Conversion Efficiency (Goals / Shots)", fontsize=10
)
axes[0].set_xlabel("Stage Reached", fontsize=10)
axes[0].grid(True, linestyle="--", alpha=0.5)

# Annotation Box for Panel A
axes[0].text(
    0.5,
    0.22,
    "Welch's t-test: t = 4.457, p < 0.0001***\nKnockout Mean: 12.5% [95% CI:"
    " 10.9%, 14.2%]\nEliminated Mean: 6.8% [95% CI: 4.7%, 8.9%]",
    transform=axes[0].transAxes,
    ha="center",
    va="top",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9
    ),
)

# Panel B: Synchronized Dual Histogram
bins = np.linspace(0, 0.25, 11)
axes[1].hist(
    df[df["Stage"] == "Knockout Stage"]["G/Sh"],
    bins=bins,
    alpha=0.6,
    color="#1f77b4",
    edgecolor="black",
    label="Knockout Qualified (n=32)",
)
axes[1].hist(
    df[df["Stage"] == "Group Stage Eliminated"]["G/Sh"],
    bins=bins,
    alpha=0.6,
    color="#ff7f0e",
    edgecolor="black",
    label="Group Eliminated (n=16)",
)
axes[1].set_title(
    "B. Goal Conversion Distribution (Shared Bins)",
    fontsize=11,
    fontweight="bold",
)
axes[1].set_xlabel(
    "Goal Conversion Efficiency (Goals / Shots)", fontsize=10
)
axes[1].set_ylabel("Number of National Teams", fontsize=10)
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("task1_high_grade_visualization.png", dpi=300)
plt.close()
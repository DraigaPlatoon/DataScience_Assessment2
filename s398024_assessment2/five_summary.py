import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("merged_fifa2026_squad_data.csv")
df["Squad"] = (
    df["Squad"]
    .astype(str)
    .str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    .str.strip()
)

plt.figure(figsize=(7, 5.5), dpi=300)
palette = {"Knockout Stage": "#2b5c8f", "Group Stage Eliminated": "#d95f02"}

# Boxplot with overlaid observations
sns.boxplot(
    x="Stage",
    y="G/Sh",
    data=df,
    palette=palette,
    width=0.36,
    boxprops=dict(alpha=0.8, edgecolor="black", linewidth=1.2),
    whiskerprops=dict(linewidth=1.2, color="black"),
    capprops=dict(linewidth=1.2, color="black"),
    medianprops=dict(linewidth=2, color="crimson"),
    showmeans=True,
    meanprops=dict(
        marker="D",
        markerfacecolor="gold",
        markeredgecolor="black",
        markersize=7,
    ),
)
sns.stripplot(
    x="Stage",
    y="G/Sh",
    data=df,
    color="black",
    alpha=0.55,
    jitter=0.18,
    size=6,
)

plt.title(
    "Goal Conversion Efficiency: Five-Number Summary & Observations",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
plt.ylabel(
    "Goal Conversion Efficiency (Gls / Sh)",
    fontsize=10,
    fontweight="semibold",
)
plt.xlabel(
    "Tournament Qualification Status", fontsize=10, fontweight="semibold"
)
plt.grid(True, linestyle=":", alpha=0.6)

# Annotate extremes
max_squad = df.loc[df["G/Sh"].idxmax()]
min_squad = df.loc[df["G/Sh"].idxmin()]
clean_name = lambda s: s.split(" ", 1)[-1] if " " in s else s

plt.annotate(
    f"Highest: {clean_name(max_squad['Squad'])}\n({max_squad['G/Sh']*100:.1f}%)",
    xy=(0, max_squad["G/Sh"]),
    xytext=(0.28, max_squad["G/Sh"] - 0.015),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    fontsize=8.5,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="#2b5c8f",
        lw=1,
    ),
)

plt.annotate(
    f"Lowest: {clean_name(min_squad['Squad'])}\n({min_squad['G/Sh']*100:.1f}%)",
    xy=(1, min_squad["G/Sh"]),
    xytext=(1.22, min_squad["G/Sh"] + 0.02),
    arrowprops=dict(arrowstyle="->", color="black", lw=1),
    fontsize=8.5,
    bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor="white",
        edgecolor="#d95f02",
        lw=1,
    ),
)

plt.tight_layout()
plt.savefig("chart1_boxplot_eda.png", dpi=300)
plt.close()
print("--> Saved 'chart1_boxplot_eda.png'!")
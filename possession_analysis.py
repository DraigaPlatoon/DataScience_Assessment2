import pandas as pd
import numpy as np
import os
from scipy import stats
import math

print("Current folder:", os.getcwd())
print("Files in this folder:", os.listdir())
print("-" * 60)

# ============================================
# 1. Load the data (using CSV - recommended)
# ============================================
try:
    df = pd.read_csv("wc2026_attacking.csv", encoding="latin1")
    print("✅ CSV file loaded successfully!")
except Exception as e:
    print("❌ Error loading file:", e)
    exit()

# ============================================
# 2. Inspect the data
# ============================================
print("\nShape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# ============================================
# 3. Create High vs Low Possession groups
# ============================================
median_possession = df["Possession_Control_%"].median()
print(f"\nMedian Possession Control: {median_possession}%")

df["Possession_Group"] = np.where(
    df["Possession_Control_%"] >= median_possession,
    "High Possession",
    "Low Possession"
)

print("\nNumber of teams in each group:")
print(df["Possession_Group"].value_counts())

print("\nSample of the data with new group column:")
print(df[["Team", "Possession_Control_%", "Attempts_On_Target", "Possession_Group"]].head(10))

# ============================================
# 4. Descriptive Statistics
# ============================================
print("\n" + "="*60)
print("DESCRIPTIVE STATISTICS")
print("="*60)

high = df[df["Possession_Group"] == "High Possession"]["Attempts_On_Target"]
low  = df[df["Possession_Group"] == "Low Possession"]["Attempts_On_Target"]

print("\nHigh Possession Teams - Attempts on Target:")
print(f"  Count  : {len(high)}")
print(f"  Mean   : {high.mean():.2f}")
print(f"  Median : {high.median():.2f}")
print(f"  Std Dev: {high.std():.2f}")
print(f"  Min    : {high.min()}")
print(f"  Max    : {high.max()}")

print("\nLow Possession Teams - Attempts on Target:")
print(f"  Count  : {len(low)}")
print(f"  Mean   : {low.mean():.2f}")
print(f"  Median : {low.median():.2f}")
print(f"  Std Dev: {low.std():.2f}")
print(f"  Min    : {low.min()}")
print(f"  Max    : {low.max()}")

# ============================================
# 5. Confidence Interval (High Possession)
# ============================================
print("\n" + "="*60)
print("95% CONFIDENCE INTERVAL (High Possession Teams)")
print("="*60)

n = len(high)
mean = high.mean()
std = high.std(ddof=1)
se = std / math.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
margin = t_crit * se

ci_low = mean - margin
ci_upp = mean + margin

print(f"Sample size (n)     : {n}")
print(f"Mean                : {mean:.2f}")
print(f"Standard Error      : {se:.2f}")
print(f"95% CI              : ({ci_low:.2f}, {ci_upp:.2f})")

# ============================================
# 6. Two-Sample t-Test
# ============================================
print("\n" + "="*60)
print("TWO-SAMPLE t-TEST")
print("="*60)
print("H0: Mean Attempts on Target of High Possession = Mean of Low Possession")
print("H1: Mean Attempts on Target of High Possession > Mean of Low Possession")
print("(one-sided test)")

t_stat, p_value = stats.ttest_ind(high, low, equal_var=False, alternative='greater')

print(f"\nt-statistic : {t_stat:.3f}")
print(f"p-value     : {p_value:.4f}")

if p_value < 0.05:
    print("\nConclusion: We REJECT the null hypothesis.")
    print("There is significant evidence that High Possession teams have higher average Attempts on Target.")
else:
    print("\nConclusion: We FAIL TO REJECT the null hypothesis.")
    print("There is not enough evidence that High Possession teams have higher average Attempts on Target.")

# ============================================
# 7. Save the cleaned dataset
# ============================================
df.to_csv("wc2026_attacking_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as 'wc2026_attacking_cleaned.csv'")
print("Analysis complete!")


# ============================================
# 8. Final Visualisations
# ============================================
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Graph 1: Boxplot with points
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Possession_Group", y="Attempts_On_Target", 
            hue="Possession_Group", palette="Set2", legend=False, width=0.5)
sns.stripplot(data=df, x="Possession_Group", y="Attempts_On_Target", 
              color="black", alpha=0.6, size=7)
plt.title("Attempts on Target by Possession Group", fontsize=14)
plt.xlabel("Possession Group")
plt.ylabel("Attempts on Target")
plt.tight_layout()
plt.savefig("01_boxplot.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Graph 1 saved: 01_boxplot.png")

# Graph 2: Bar chart
plt.figure(figsize=(8, 6))
means = df.groupby("Possession_Group")["Attempts_On_Target"].mean().reset_index()
ax = sns.barplot(data=means, x="Possession_Group", y="Attempts_On_Target",
                 hue="Possession_Group", palette="Set2", legend=False, edgecolor="black")
plt.title("Average Attempts on Target by Group", fontsize=14)
plt.ylabel("Average Attempts on Target")
plt.xlabel("Possession Group")

for i, row in means.iterrows():
    ax.text(i, row["Attempts_On_Target"] + 0.8, f"{row['Attempts_On_Target']:.1f}",
            ha="center", fontsize=13, fontweight="bold")

plt.ylim(0, 35)
plt.tight_layout()
plt.savefig("02_barchart.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Graph 2 saved: 02_barchart.png")

# Graph 3: Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Possession_Control_%", y="Attempts_On_Target",
                hue="Possession_Group", palette="Set2", s=110, edgecolor="black")
plt.title("Possession % vs Attempts on Target", fontsize=14)
plt.xlabel("Possession Control (%)")
plt.ylabel("Attempts on Target")
plt.tight_layout()
plt.savefig("03_scatter.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Graph 3 saved: 03_scatter.png")

print("\n All 3 graphs saved successfully!")


# ============================================
# Extra Charts: Confidence Interval & Welch's t-test
# ============================================
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_style("whitegrid")

# ---------- Chart 1: Confidence Interval Visualisation ----------
plt.figure(figsize=(8, 5))

# High Possession mean and CI
mean_high = high.mean()
ci_low = 21.42
ci_upp = 32.50

plt.errorbar(x=["High Possession"], y=[mean_high], 
             yerr=[[mean_high - ci_low], [ci_upp - mean_high]],
             fmt='o', color='#66c2a5', markersize=12, capsize=10, capthick=2, elinewidth=2)

plt.axhline(y=mean_high, color='#66c2a5', linestyle='--', alpha=0.5)
plt.title("95% Confidence Interval\nHigh Possession Teams - Attempts on Target", fontsize=14)
plt.ylabel("Attempts on Target", fontsize=12)
plt.ylim(15, 40)

# Add text labels
plt.text(0, mean_high + 1.5, f"Mean = {mean_high:.1f}", ha='center', fontsize=12, fontweight='bold')
plt.text(0, ci_low - 1.5, f"95% CI: ({ci_low:.1f}, {ci_upp:.1f})", ha='center', fontsize=11)

plt.tight_layout()
plt.savefig("04_confidence_interval.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 4 saved: 04_confidence_interval.png")


# ---------- Chart 2: Welch's t-test Mean Comparison ----------
plt.figure(figsize=(9, 6))

groups = ["High Possession", "Low Possession"]
means = [high.mean(), low.mean()]
errors = [high.sem(), low.sem()]   # Standard Error

bars = plt.bar(groups, means, yerr=errors, capsize=10, 
               color=["#66c2a5", "#fc8d62"], edgecolor="black", width=0.6)

plt.title("Welch's t-test Result\nHigh vs Low Possession Teams", fontsize=14)
plt.ylabel("Average Attempts on Target", fontsize=12)
plt.ylim(0, 35)

# Add mean values on bars
for bar, mean in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width()/2, mean + 1.2, f"{mean:.1f}", 
             ha='center', fontsize=13, fontweight='bold')

# Add p-value and conclusion
plt.text(0.5, 30, "Welch's t-test\nt = 6.22, p < 0.001\nSignificant difference", 
         ha='center', fontsize=12, bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

plt.tight_layout()
plt.savefig("05_welch_ttest.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ Chart 5 saved: 05_welch_ttest.png")

print("\n Extra charts created successfully!")

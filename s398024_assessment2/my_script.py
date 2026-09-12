import math
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import statsmodels.stats.weightstats as stm

print("=== STEP 1: LOADING & MERGING DATA ===")

# Load datasets
df_squad = pd.read_csv("squad_shooting.csv", header=1)
df_player = pd.read_csv("player_shooting.csv", header=1)

print("Original squad rows: %d, cols: %d" % (df_squad.shape[0], df_squad.shape[1]))
print("Original player rows: %d, cols: %d" % (df_player.shape[0], df_player.shape[1]))

# Clean Squad names
df_squad["Squad"] = (
    df_squad["Squad"]
    .astype(str)
    .str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    .str.replace("\xa0", " ")
    .str.strip()
)
df_player["Squad"] = (
    df_player["Squad"]
    .astype(str)
    .str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    .str.replace("\xa0", " ")
    .str.strip()
)

# Convert numeric columns
numeric_cols = ["90s", "Gls", "Sh", "SoT", "G/Sh"]
for col in numeric_cols:
    df_squad[col] = pd.to_numeric(df_squad[col], errors="coerce")

df_squad = df_squad.dropna(subset=["Squad", "G/Sh"]).copy()

# Add Stage indicator (Knockout > 3.0 matches vs Group Eliminated <= 3.0 matches)
df_squad["Stage"] = df_squad["90s"].apply(
    lambda x: "Knockout Stage" if x > 3.0 else "Group Stage Eliminated"
)

# Clean Player dataset and aggregate squad-level metrics
df_player_clean = df_player[df_player["Player"] != "Player"].copy()
df_player_clean["Age"] = pd.to_numeric(df_player_clean["Age"], errors="coerce")

player_summary = (
    df_player_clean.groupby("Squad")
    .agg(Avg_Squad_Age=("Age", "mean"), Total_Players_Used=("Player", "count"))
    .reset_index()
)

# Merge squad with player summary on 'Squad'
df_merged = pd.merge(df_squad, player_summary, on="Squad", how="left")

# Save merged dataset to disk
output_csv = "merged_fifa2026_squad_data.csv"
df_merged.to_csv(output_csv, index=False)
print("--> Successfully created and saved: '%s' (%d rows, %d cols)\n" % (
    output_csv,
    df_merged.shape[0],
    df_merged.shape[1],
))


print("=== STEP 2: DESCRIPTIVE STATISTICS ===")

df_ko = df_merged[df_merged["Stage"] == "Knockout Stage"]
df_elim = df_merged[df_merged["Stage"] == "Group Stage Eliminated"]

sample_ko = df_ko["G/Sh"].to_numpy()
sample_elim = df_elim["G/Sh"].to_numpy()

print("Knockout Stage (G/Sh) Summary:")
print(df_ko["G/Sh"].describe())

print("\nGroup Stage Eliminated (G/Sh) Summary:")
print(df_elim["G/Sh"].describe())

# Save distribution histogram
plt.figure(figsize=(8, 4.5))
plt.hist(
    sample_ko,
    bins=8,
    alpha=0.6,
    color="blue",
    edgecolor="black",
    label="Knockout Qualified (n=32)",
)
plt.hist(
    sample_elim,
    bins=8,
    alpha=0.6,
    color="orange",
    edgecolor="black",
    label="Group Eliminated (n=16)",
)
plt.title("Goal Conversion Efficiency (G/Sh) Distribution")
plt.xlabel("Goal Conversion Efficiency (G/Sh)")
plt.ylabel("Number of Teams")
plt.legend()
plt.tight_layout()
chart_file = "task1_conversion_dist.png"
plt.savefig(chart_file, dpi=300)
plt.close()
print("--> Successfully created and saved chart: '%s'\n" % chart_file)


print("=== STEP 3: CONFIDENCE INTERVALS ===")

x_bar1, s1, n1 = st.tmean(sample_ko), st.tstd(sample_ko), len(sample_ko)
x_bar2, s2, n2 = st.tmean(sample_elim), st.tstd(sample_elim), len(sample_elim)

# Knockout (n = 32 >= 30, large sample)
ci_low_ko, ci_upp_ko = stm._tconfint_generic(
    x_bar1, s1 / math.sqrt(n1), n1 - 1, alpha=0.05, alternative="two-sided"
)

# Eliminated (n = 16 < 30, small sample -> requires t*)
ci_low_elim, ci_upp_elim = stm._tconfint_generic(
    x_bar2, s2 / math.sqrt(n2), n2 - 1, alpha=0.05, alternative="two-sided"
)

print("Knockout Stage (n=%d): Mean = %.4f, 95%% CI = [%.4f, %.4f]" % (n1, x_bar1, ci_low_ko, ci_upp_ko))
print("Group Eliminated (n=%d): Mean = %.4f, 95%% CI = [%.4f, %.4f]\n" % (n2, x_bar2, ci_low_elim, ci_upp_elim))


print("=== STEP 4: TWO-SAMPLE t-TEST ===")
# H0: mu_ko = mu_elim  vs  Ha: mu_ko > mu_elim (one-sided)
t_stat, p_val = st.ttest_ind_from_stats(
    x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative="greater"
)

print("t-statistic (t*): %.4f" % t_stat)
print("p-value: %.6f" % p_val)
if p_val < 0.05:
    print(
        "\nConclusion: Reject H0 at alpha = 0.05. Knockout Stage teams achieved a significantly higher Goal Conversion Efficiency."
    )
else:
    print("\nConclusion: Fail to reject H0.")

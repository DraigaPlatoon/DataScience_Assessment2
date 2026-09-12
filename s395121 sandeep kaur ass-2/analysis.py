import pandas as pd

# Read the FBref CSV using its two header rows
df = pd.read_csv(
    "world_cup_2026_player_stats.csv",
    header=[0, 1]
)

# Keep the useful second-level column names
df.columns = [
    "Rk", "Player", "Pos", "Squad", "Age", "Club", "Born",
    "MP", "Starts", "Min", "90s",
    "Gls", "Ast", "G+A", "G-PK", "PK", "PKatt", "CrdY", "CrdR",
    "Gls_per90", "Ast_per90", "G+A_per90", "G-PK_per90",
    "G+A-PK_per90", "Matches", "Player_ID"
]

print(df.head())
print("\nDataset size:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
# Select important variables for our analysis
important_columns = ["Player", "Pos", "Squad", "Age", "MP", "Starts", "Min", "Gls", "Ast", "G+A"]

data = df[important_columns].copy()

print("\nImportant variables:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())

print("\nData types:")
print(data.dtypes)

# --------------------------------------------------
# DEFINE ELIGIBLE PLAYERS AND PLAYING ROLE
# --------------------------------------------------

# Use players who played at least 90 minutes.
# This avoids unstable per-90 rates from players with very little playing time.
analysis_data = df[df["Min"] >= 90].copy()

# Calculate the proportion of appearances that each player started.
analysis_data["Start_Rate"] = (
    analysis_data["Starts"] / analysis_data["MP"]
)

# Classify players into two groups.
analysis_data["Role"] = analysis_data["Start_Rate"].apply(
    lambda x: "Regular Starter" if x >= 0.50 else "Primarily Substitute"
)

print("\nNumber of eligible players:", len(analysis_data))

print("\nPlayers in each role:")
print(analysis_data["Role"].value_counts())

print("\nExample classified players:")
print(
    analysis_data[
        ["Player", "MP", "Starts", "Min",
         "Start_Rate", "Role", "G+A_per90"]
    ].head(10)
)


# --------------------------------------------------
# RANDOM SAMPLING
# --------------------------------------------------

# Separate the two populations
starters = analysis_data[
    analysis_data["Role"] == "Regular Starter"
]

substitutes = analysis_data[
    analysis_data["Role"] == "Primarily Substitute"
]

# Randomly select 60 players from each group.
# random_state=42 makes the sample reproducible.
starter_sample = starters.sample(
    n=60,
    random_state=42
)

substitute_sample = substitutes.sample(
    n=60,
    random_state=42
)

print("\nSample sizes:")
print("Regular Starter sample:", len(starter_sample))
print("Primarily Substitute sample:", len(substitute_sample))

print("\nFirst 5 sampled Regular Starters:")
print(
    starter_sample[
        ["Player", "Min", "Starts", "MP", "G+A_per90"]
    ].head()
)

print("\nFirst 5 sampled Primarily Substitutes:")
print(
    substitute_sample[
        ["Player", "Min", "Starts", "MP", "G+A_per90"]
    ].head()
)


# --------------------------------------------------
# DESCRIPTIVE STATISTICS
# --------------------------------------------------

starter_stats = starter_sample["G+A_per90"].describe()
substitute_stats = substitute_sample["G+A_per90"].describe()

print("\nDescriptive statistics - Regular Starters:")
print(starter_stats)

print("\nDescriptive statistics - Primarily Substitutes:")
print(substitute_stats)

# Additional useful statistics
print("\nRegular Starter median:",
      starter_sample["G+A_per90"].median())

print("Primarily Substitute median:",
      substitute_sample["G+A_per90"].median())

print("\nRegular Starter variance:",
      starter_sample["G+A_per90"].var(ddof=1))

print("Primarily Substitute variance:",
      substitute_sample["G+A_per90"].var(ddof=1))


# --------------------------------------------------
# 95% CONFIDENCE INTERVALS
# --------------------------------------------------

import math

# 95% critical z-value
z_star = 1.96

# Regular Starters
starter_mean = starter_sample["G+A_per90"].mean()
starter_std = starter_sample["G+A_per90"].std(ddof=1)
starter_n = len(starter_sample)

starter_se = starter_std / math.sqrt(starter_n)
starter_margin = z_star * starter_se

starter_ci_lower = starter_mean - starter_margin
starter_ci_upper = starter_mean + starter_margin

# Primarily Substitutes
sub_mean = substitute_sample["G+A_per90"].mean()
sub_std = substitute_sample["G+A_per90"].std(ddof=1)
sub_n = len(substitute_sample)

sub_se = sub_std / math.sqrt(sub_n)
sub_margin = z_star * sub_se

sub_ci_lower = sub_mean - sub_margin
sub_ci_upper = sub_mean + sub_margin

print("\n95% Confidence Interval - Regular Starters:")
print("Mean:", round(starter_mean, 3))
print("Standard Error:", round(starter_se, 3))
print("95% CI:",
      (round(starter_ci_lower, 3),
       round(starter_ci_upper, 3)))

print("\n95% Confidence Interval - Primarily Substitutes:")
print("Mean:", round(sub_mean, 3))
print("Standard Error:", round(sub_se, 3))
print("95% CI:",
      (round(sub_ci_lower, 3),
       round(sub_ci_upper, 3)))

# --------------------------------------------------
# TWO-SAMPLE T-TEST
# --------------------------------------------------

from scipy import stats

# Significance level
alpha = 0.05

# Perform an independent two-sample t-test
t_stat, p_value = stats.ttest_ind(
    starter_sample["G+A_per90"],
    substitute_sample["G+A_per90"],
    equal_var=False
)

print("\nTwo-Sample T-Test")
print("H0: Mean G+A per 90 is equal for the two groups.")
print("H1: Mean G+A per 90 is different between the two groups.")
print("Significance level (alpha):", alpha)
print("t-statistic:", round(t_stat, 3))
print("p-value:", round(p_value, 4))

if p_value < alpha:
    print("Decision: Reject H0.")
    print("Conclusion: There is a statistically significant difference in mean G+A per 90 between the two groups.")
else:
    print("Decision: Fail to reject H0.")
    print("Conclusion: There is insufficient evidence of a statistically significant difference in mean G+A per 90 between the two groups.")

    # --------------------------------------------------
# VISUALISE THE DISTRIBUTIONS
# --------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.hist(
    starter_sample["G+A_per90"],
    bins=10,
    alpha=0.6,
    label="Regular Starters"
)

plt.hist(
    substitute_sample["G+A_per90"],
    bins=10,
    alpha=0.6,
    label="Primarily Substitutes"
)

plt.xlabel("Goals + Assists per 90 minutes")
plt.ylabel("Number of Players")
plt.title("Distribution of G+A per 90 by Playing Role")
plt.legend()

plt.tight_layout()

plt.savefig("ga_per90_distribution.png", dpi=300)

plt.close()


# --------------------------------------------------
# FINAL RESULTS GRAPH: MEANS WITH 95% CI
# --------------------------------------------------

groups = ["Regular Starters", "Primarily Substitutes"]

means = [
    starter_mean,
    sub_mean
]

margins = [
    starter_margin,
    sub_margin
]

plt.figure(figsize=(8, 5))

plt.bar(
    groups,
    means,
    yerr=margins,
    capsize=8
)

plt.ylabel("Mean Goals + Assists per 90")
plt.title("Mean G+A per 90 with 95% Confidence Intervals")

plt.tight_layout()

plt.savefig(
    "ga_per90_mean_ci.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show() 

# --------------------------------------------------
# BOXPLOT: G+A PER 90 BY PLAYING ROLE
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    [
        starter_sample["G+A_per90"],
        substitute_sample["G+A_per90"]
    ],
    labels=["Regular Starters", "Primarily Substitutes"]
)

plt.ylabel("Goals + Assists per 90 minutes")
plt.title("Boxplot of G+A per 90 by Playing Role")

plt.tight_layout()

plt.savefig(
    "ga_per90_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Boxplot saved successfully.")
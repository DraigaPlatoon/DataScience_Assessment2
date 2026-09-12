import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

# 1. Dataset of all 48 qualified nations
df = pd.read_csv("datasheet.txt")

# 2. Split cohorts
uefa = df[df["Confederation"] == "UEFA"]["Shots_per_90"]
non_uefa = df[df["Confederation"] == "Non-UEFA"]["Shots_per_90"]

# 3. Descriptive Statistics
uefa_mean, uefa_std, uefa_n = uefa.mean(), uefa.std(), len(uefa)
non_mean, non_std, non_n = non_uefa.mean(), non_uefa.std(), len(non_uefa)

print(f"UEFA: n={uefa_n}, Mean={uefa_mean:.2f}, Std={uefa_std:.2f}")
print(f"Non-UEFA: n={non_n}, Mean={non_mean:.2f}, Std={non_std:.2f}")

# 4. Inferential Statistics: 95% Confidence Intervals (t-distribution)
ci_uefa = st.t.interval(0.95, df=uefa_n - 1, loc=uefa_mean, scale=st.sem(uefa))
ci_non = st.t.interval(0.95, df=non_n - 1, loc=non_mean, scale=st.sem(non_uefa))

print(f"UEFA 95% CI: [{ci_uefa[0]:.2f}, {ci_uefa[1]:.2f}]")
print(f"Non-UEFA 95% CI: [{ci_non[0]:.2f}, {ci_non[1]:.2f}]")

# 5. Inferential Statistics: Two-Sample t-Test (Independent)
t_stat, p_val_two = st.ttest_ind(uefa, non_uefa, equal_var=True)
p_val_one = p_val_two / 2 if t_stat > 0 else 1 - (p_val_two / 2)

print(f"t-statistic: {t_stat:.4f}")
print(f"One-tailed p-value: {p_val_one:.8e}")

# 6. Generate Boxplot
plt.figure(figsize=(7, 5))
bp = plt.boxplot(
    [uefa, non_uefa],
    labels=["UEFA Teams\n(n=16)", "Non-UEFA Teams\n(n=32)"],
    patch_artist=True,
)
bp["boxes"][0].set_facecolor("#4A90E2")
bp["boxes"][1].set_facecolor("#F5A623")
plt.title(
    "FIFA World Cup 2026: Shots per 90 Minutes\nUEFA vs. Non-UEFA Confederations",
    fontweight="bold",
)
plt.ylabel("Shots per 90 Minutes (Shot/90)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig("uefa_vs_non_uefa_shots.png", dpi=300)
plt.close()

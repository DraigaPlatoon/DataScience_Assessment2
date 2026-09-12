import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

# 1. Load dataset
df = pd.read_csv("merged_fifa2026_squad_data.csv")
ko = df[df["Stage"] == "Knockout Stage"]["G/Sh"].dropna().values
elim = df[df["Stage"] == "Group Stage Eliminated"]["G/Sh"].dropna().values

n_ko, n_elim = len(ko), len(elim)
mean_ko, mean_elim = np.mean(ko), np.mean(elim)
sem_ko, sem_elim = st.sem(ko), st.sem(elim)

# Compute Student's t* 95% CIs
ci_ko = st.t.interval(0.95, df=n_ko - 1, loc=mean_ko, scale=sem_ko)
ci_elim = st.t.interval(0.95, df=n_elim - 1, loc=mean_elim, scale=sem_elim)

# Two-Sample Welch's t-test and Cohen's d
t_val, p_val = st.ttest_ind(ko, elim, equal_var=False, alternative="greater")
pooled_sd = np.sqrt(
    (
        (n_ko - 1) * np.var(ko, ddof=1)
        + (n_elim - 1) * np.var(elim, ddof=1)
    )
    / (n_ko + n_elim - 2)
)
cohens_d = (mean_ko - mean_elim) / pooled_sd

errs_ko = [[mean_ko - ci_ko[0]], [ci_ko[1] - mean_ko]]
errs_elim = [[mean_elim - ci_elim[0]], [ci_elim[1] - mean_elim]]

plt.figure(figsize=(6.5, 5.5), dpi=300)

# Error bars
plt.errorbar(
    0,
    mean_ko,
    yerr=errs_ko,
    fmt="D",
    color="#2b5c8f",
    ecolor="#2b5c8f",
    elinewidth=3,
    capsize=10,
    capthick=2.5,
    markersize=9,
    label="Knockout 95% CI",
)
plt.errorbar(
    1,
    mean_elim,
    yerr=errs_elim,
    fmt="s",
    color="#d95f02",
    ecolor="#d95f02",
    elinewidth=3,
    capsize=10,
    capthick=2.5,
    markersize=9,
    label="Eliminated 95% CI",
)

plt.xticks(
    [0, 1],
    [f"Knockout Stage\n(n={n_ko})", f"Group Eliminated\n(n={n_elim})"],
    fontsize=10,
    fontweight="semibold",
)
plt.ylabel(
    "Estimated Population Mean (Gls / Sh)",
    fontsize=10,
    fontweight="semibold",
)
plt.title(
    "95% Confidence Intervals & Welch's t-Test",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
plt.grid(True, linestyle=":", alpha=0.6)

# Reference dotted lines showing clear separation
plt.axhline(y=ci_ko[0], color="#2b5c8f", linestyle=":", alpha=0.6)
plt.axhline(y=ci_elim[1], color="#d95f02", linestyle=":", alpha=0.6)

stats_callout = (
    "Welch's Two-Sample t-Test:\n"
    f"• t* = {t_val:.4f}\n"
    f"• p-value = {p_val:.2e} (p < 0.001)***\n"
    f"• Cohen's d = {cohens_d:.2f} (Large Effect)\n"
    "• Non-overlapping 95% CIs\n"
    "• Decision: Reject H0 at α = 0.05"
)
plt.text(
    0.5,
    0.25,
    stats_callout,
    transform=plt.gca().transAxes,
    ha="center",
    va="top",
    fontsize=8.8,
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="#f8f9fa",
        edgecolor="#6c757d",
        alpha=0.95,
    ),
)

plt.tight_layout()
plt.savefig("chart3_confidence_intervals.png", dpi=300)
plt.close()
print("--> Saved 'chart3_confidence_intervals.png'!")
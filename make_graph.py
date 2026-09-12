import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

# Results from our analysis
groups = ["Regular Starters", "Primarily Substitutes"]
means = [0.175, 0.330]
errors = [0.094, 0.153]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    groups,
    means,
    yerr=errors,
    capsize=8
)

plt.ylabel("Mean Goals + Assists per 90")
plt.title("Mean G+A per 90 with 95% Confidence Intervals")

plt.text(0, 0.19, "0.175", ha="center")
plt.text(1, 0.345, "0.330", ha="center")

plt.tight_layout()

output_file = Path.cwd() / "ga_per90_mean_ci.png"

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Graph saved successfully.")
print("Location:", output_file)
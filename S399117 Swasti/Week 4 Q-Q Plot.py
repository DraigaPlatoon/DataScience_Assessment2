import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

# Load dataset
df = pd.read_csv("./datasheet.csv")

# Generate Q-Q plot against a normal distribution
plt.figure(figsize=(7, 5))
st.probplot(df["Shots_per_90"], dist="norm", plot=plt)

plt.title(
    "Week 4: Q-Q Plot (t-Test Normality Assumption)",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Ordered Values (Shots/90)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("chart3_qqplot.png", dpi=300)
plt.show()

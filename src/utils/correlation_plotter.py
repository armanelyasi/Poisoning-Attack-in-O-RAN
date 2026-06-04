import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Parse data
df = pd.read_csv('/datasets/corr_matrx.csv', sep="\t", index_col=0)
df = df.drop(index='reward', columns='reward')

# Plot heatmap
plt.figure(figsize=(6.4, 6.4))  # Size to fit two-column paper
sns.heatmap(df, annot=False, cmap='bwr', cbar=True, square=True, linewidths=0.5)

# Save to PDF
plt.tight_layout()
plt.savefig("out/correlation_matrix.pdf", dpi=300)

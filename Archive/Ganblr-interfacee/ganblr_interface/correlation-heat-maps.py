import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Argument parser
parser = argparse.ArgumentParser(description="Generate correlation heatmaps.")
parser.add_argument("--real", required=True, help="Path to the real dataset file.")
parser.add_argument("--synthetic", required=True, help="Path to the synthetic dataset file.")
parser.add_argument("--output", required=True, help="Directory to save the output plots.")
args = parser.parse_args()

# Load datasets
real_data = pd.read_csv(args.real)
synthetic_data = pd.read_csv(args.synthetic)

# Compute correlations
real_corr = real_data.corr()
synthetic_corr = synthetic_data.corr()

# Plot real data heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(real_corr, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap - Real Data")
real_output_path = os.path.join(args.output, "real_data_correlation_heatmap.png")
plt.savefig(real_output_path)
plt.close()

# Plot synthetic data heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(synthetic_corr, annot=False, cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap - Synthetic Data")
synthetic_output_path = os.path.join(args.output, "synthetic_data_correlation_heatmap.png")
plt.savefig(synthetic_output_path)
plt.close()

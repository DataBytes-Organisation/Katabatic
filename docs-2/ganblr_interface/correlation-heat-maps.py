import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

# Argument parser for output directory
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="Path to the folder to save output images")
args = parser.parse_args()
output_dir = args.output

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load datasets
real_data = pd.read_csv('credit_X_train.csv')  # File for real data
synthetic_data = pd.read_csv('credit_synthetic_data.csv')  # File for synthetic data

# Ensure column names match
if list(real_data.columns) != list(synthetic_data.columns):
    synthetic_data.columns = real_data.columns

# Compute correlations for both datasets
real_corr = real_data.corr()
synthetic_corr = synthetic_data.corr()

# Plot Real Data Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(real_corr, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap - Real Data')
real_heatmap_path = os.path.join(output_dir, 'real_data_correlation_heatmap.png')
plt.savefig(real_heatmap_path)
plt.close()
print(f"Real data heatmap saved to {real_heatmap_path}")

# Plot Synthetic Data Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(synthetic_corr, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap - Synthetic Data')
synthetic_heatmap_path = os.path.join(output_dir, 'synthetic_data_correlation_heatmap.png')
plt.savefig(synthetic_heatmap_path)
plt.close()
print(f"Synthetic data heatmap saved to {synthetic_heatmap_path}")

# Save correlation matrices to CSV files for further analysis
real_corr_path = os.path.join(output_dir, 'real_data_correlation_matrix.csv')
synthetic_corr_path = os.path.join(output_dir, 'synthetic_data_correlation_matrix.csv')
real_corr.to_csv(real_corr_path, index=True)
synthetic_corr.to_csv(synthetic_corr_path, index=True)
print(f"Real data correlation matrix saved to {real_corr_path}")
print(f"Synthetic data correlation matrix saved to {synthetic_corr_path}")

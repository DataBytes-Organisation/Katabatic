import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load datasets
real_data = pd.read_csv('credit_X_train.csv')  # Replace with your real data file
synthetic_data = pd.read_csv('credit_synthetic_data.csv')  # Replace with your synthetic data file

# Ensure column names match
if list(real_data.columns) != list(synthetic_data.columns):
    synthetic_data.columns = real_data.columns

# Compute correlations
real_corr = real_data.corr()
synthetic_corr = synthetic_data.corr()

# Plot heatmaps
plt.figure(figsize=(10, 8))
sns.heatmap(real_corr, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap - Real Data')
plt.savefig('real_data_correlation_heatmap.png')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(synthetic_corr, annot=False, cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap - Synthetic Data')
plt.savefig('synthetic_data_correlation_heatmap.png')
plt.show()

# Save correlation matrices to CSV for documentation purposes
real_corr.to_csv('real_data_correlation_matrix.csv')
synthetic_corr.to_csv('synthetic_data_correlation_matrix.csv')

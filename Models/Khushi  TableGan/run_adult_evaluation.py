import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eval import plot_var_cor, plot_corr_diff, plot_dim_red

# Load the Adult dataset
print("Loading Adult dataset...")
adult_data = pd.read_csv('data/Adult/train_Adult_cleaned.csv')
adult_labels = pd.read_csv('data/Adult/train_Adult_labels.csv')

# Display basic info about the dataset
print(f"Adult dataset shape: {adult_data.shape}")
print(f"Adult labels shape: {adult_labels.shape}")
print("\nSample of Adult dataset:")
print(adult_data.head())

# Let's create a subset of the data for evaluation
# In a real scenario, we would have both real and synthetic data to compare
# For demonstration, we'll split the real data into two parts

# Split the data
np.random.seed(42)
msk = np.random.rand(len(adult_data)) < 0.7
real_data = adult_data[msk]
synthetic_data = adult_data[~msk]  # Using a subset of real data as "synthetic" for demonstration

print(f"\nReal data shape: {real_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

# Plot correlation matrix for the real data
plt.figure(figsize=(12, 10))
print("\nGenerating correlation plot for real data...")
plot_var_cor(real_data)
plt.title("Correlation Matrix of Adult Dataset (Real Data)")
plt.savefig("adult_real_correlation.png")
plt.close()

# Plot correlation matrix for the synthetic data
plt.figure(figsize=(12, 10))
print("\nGenerating correlation plot for synthetic data...")
plot_var_cor(synthetic_data)
plt.title("Correlation Matrix of Adult Dataset (Synthetic Data)")
plt.savefig("adult_synthetic_correlation.png")
plt.close()

# Plot correlation difference between real and synthetic data
print("\nGenerating correlation difference plot...")
plot_corr_diff(real_data, synthetic_data, plot_diff=True)
plt.suptitle("Correlation Comparison between Real and Synthetic Adult Data")
plt.savefig("adult_correlation_comparison.png")
plt.close()

# Dimensionality reduction visualization
print("\nGenerating dimensionality reduction visualization...")
plot_dim_red(adult_data)
plt.title("Dimensionality Reduction of Adult Dataset")
plt.savefig("adult_dim_reduction.png")
plt.close()

print("\nEvaluation complete. Check the generated PNG files for visualization results.") 
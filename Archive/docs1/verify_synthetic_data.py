import pandas as pd

# Load synthetic data
synthetic_data = pd.read_csv("car_synthetic_data.csv")

# Inspect first few rows
print("First few rows of the synthetic dataset:")
print(synthetic_data.head())

# Summary statistics
print("\nSummary statistics of the synthetic dataset:")
print(synthetic_data.describe())

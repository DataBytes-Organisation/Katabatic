import pandas as pd

# Load synthetic data
synthetic_data = pd.read_csv('credit_synthetic_data.csv')

# Display first few rows
print("First few rows of the synthetic data:")
print(synthetic_data.head())

# Display summary statistics
print("\nSummary statistics of the synthetic data:")
print(synthetic_data.describe())

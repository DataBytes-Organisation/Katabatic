import pandas as pd

# Load the dataset
file_path = 'creditcard.csv'  # Update with the correct path if necessary
credit_data = pd.read_csv(file_path)

# Display the first few rows
print("First few rows of the dataset:")
print(credit_data.head())

# Display dataset information
print("\nDataset Info:")
print(credit_data.info())

# Display summary statistics
print("\nSummary statistics of numerical columns:")
print(credit_data.describe())

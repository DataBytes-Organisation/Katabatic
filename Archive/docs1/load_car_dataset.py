import pandas as pd

# Load the dataset
file_path = "car.data"  # Replace with the actual path
columns = ["Buying", "Maint", "Doors", "Persons", "Lug_boot", "Safety", "Class"]
df = pd.read_csv(file_path, header=None, names=columns)

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Check the data types and for missing values
print("\nDataset Info:")
print(df.info())

# Display unique values for each column
print("\nUnique values per column:")
for column in df.columns:
    print(f"{column}: {df[column].unique()}")

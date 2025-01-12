import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "car.data"  # Replace with your actual file path
columns = ["Buying", "Maint", "Doors", "Persons", "Lug_boot", "Safety", "Class"]
df = pd.read_csv(file_path, header=None, names=columns)

# Display the first few rows
print("First few rows of the original dataset:")
print(df.head())

# Check the data types and for missing values
print("\nDataset Info:")
print(df.info())

# Display unique values for each column
print("\nUnique values per column (before encoding):")
for column in df.columns:
    print(f"{column}: {df[column].unique()}")

# Encode categorical features using LabelEncoder
encoder = LabelEncoder()
for column in df.columns:
    df[column] = encoder.fit_transform(df[column])

# Display the first few rows after encoding
print("\nDataset after encoding:")
print(df.head())

# Split the data into features (X) and target (y)
X = df.drop("Class", axis=1)  # Features
y = df["Class"]  # Target

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# Save the preprocessed dataset (optional)
df.to_csv("car_dataset_encoded.csv", index=False)
print("\nEncoded dataset saved as 'car_dataset_encoded.csv'.")

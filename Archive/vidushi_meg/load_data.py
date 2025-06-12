import pandas as pd

# Load the data
data = pd.read_csv('Katabatic/data/Adult/train_Adult_cleaned.csv')
labels = pd.read_csv('Katabatic/data/Adult/train_Adult_labels.csv')

# Check the first few rows of the data
print(data.head())
print(labels.head())

# Standardize numerical features
from sklearn.preprocessing import StandardScaler, LabelEncoder

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Encode categorical features (if any)
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

print(f"Scaled data: {data_scaled[:5]}")

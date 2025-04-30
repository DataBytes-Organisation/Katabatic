import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the datasets
data = pd.read_csv('Katabatic/data/Adult/train_Adult_cleaned.csv')
labels = pd.read_csv('Katabatic/data/Adult/train_Adult_labels.csv')

# Fill missing values with the median (for simplicity)
data = data.fillna(data.median())

# One-Hot Encoding for categorical variables
data = pd.get_dummies(data, drop_first=True)

# Standardizing the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Convert the scaled data back into a DataFrame
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

# Save the preprocessed data
X_train.to_csv('Katabatic/data/Adult/X_train.csv', index=False)
X_test.to_csv('Katabatic/data/Adult/X_test.csv', index=False)
y_train.to_csv('Katabatic/data/Adult/y_train.csv', index=False)
y_test.to_csv('Katabatic/data/Adult/y_test.csv', index=False)

# Optionally, save all preprocessed data together
data_scaled.to_csv('Katabatic/data/Adult/preprocessed_data.csv', index=False)

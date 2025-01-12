import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
file_path = 'creditcard.csv'  # Ensure this path is correct
credit_data = pd.read_csv(file_path)

# Normalize 'Amount' and 'Time'
scaler = MinMaxScaler()
credit_data[['Amount', 'Time']] = scaler.fit_transform(credit_data[['Amount', 'Time']])

# Split the dataset into features (X) and target (y)
X = credit_data.drop(columns=['Class'])
y = credit_data['Class']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed datasets
X_train.to_csv('credit_X_train.csv', index=False)
X_test.to_csv('credit_X_test.csv', index=False)
y_train.to_csv('credit_y_train.csv', index=False)
y_test.to_csv('credit_y_test.csv', index=False)

print("Preprocessing complete. Training and testing datasets saved.")

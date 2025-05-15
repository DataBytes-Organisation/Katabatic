import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load training and testing data
X_train = pd.read_csv('credit_X_train.csv')
X_test = pd.read_csv('credit_X_test.csv')
y_train = pd.read_csv('credit_y_train.csv')
y_test = pd.read_csv('credit_y_test.csv')

# Ensure y_train and y_test are in the correct shape
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Load synthetic data
synthetic_data = pd.read_csv('credit_synthetic_data.csv')

# Split features and labels from synthetic data
X_synthetic = synthetic_data.iloc[:, :-1]  # All columns except the last
y_synthetic = synthetic_data.iloc[:, -1]   # The last column as labels

# Fix synthetic labels (convert to discrete classes if needed)
if y_synthetic.dtype != int:
    y_synthetic = (y_synthetic > 0.5).astype(int)  # Convert to binary labels (0 or 1)

# Align columns in X_synthetic and X_test to match X_train
missing_columns = set(X_train.columns) - set(X_synthetic.columns)
extra_columns = set(X_synthetic.columns) - set(X_train.columns)

# Add missing columns to X_synthetic with default values
for col in missing_columns:
    X_synthetic[col] = 0  # You can replace 0 with other default values if needed

# Drop extra columns from X_synthetic
X_synthetic = X_synthetic.drop(columns=extra_columns, errors='ignore')

# Ensure the column order in X_synthetic matches X_train
X_synthetic = X_synthetic[X_train.columns]

# Repeat the process for X_test
missing_columns = set(X_train.columns) - set(X_test.columns)
extra_columns = set(X_test.columns) - set(X_train.columns)

for col in missing_columns:
    X_test[col] = 0  # Add missing columns with default values

X_test = X_test.drop(columns=extra_columns, errors='ignore')
X_test = X_test[X_train.columns]

# Train on synthetic data and test on real data (TSTR)
rf_tstr = RandomForestClassifier(random_state=42)
rf_tstr.fit(X_synthetic, y_synthetic)  # Train on synthetic with its labels
y_pred_tstr = rf_tstr.predict(X_test)  # Test on real
tstr_accuracy = accuracy_score(y_test, y_pred_tstr)

# Train on real data and test on real data (TRTR)
rf_trtr = RandomForestClassifier(random_state=42)
rf_trtr.fit(X_train, y_train)  # Train on real
y_pred_trtr = rf_trtr.predict(X_test)  # Test on real
trtr_accuracy = accuracy_score(y_test, y_pred_trtr)

# Print results
print(f"TSTR Accuracy (Synthetic -> Real): {tstr_accuracy:.4f}")
print(f"TRTR Accuracy (Real -> Real): {trtr_accuracy:.4f}")

# Save results to a CSV file
results = {
    "Metric": ["TSTR Accuracy", "TRTR Accuracy"],
    "Accuracy": [tstr_accuracy, trtr_accuracy]
}
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)
print("Evaluation results saved to 'evaluation_results.csv'.")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the datasets
real_data = pd.read_csv('car_dataset_encoded.csv')
synthetic_data = pd.read_csv('car_synthetic_data.csv')

# Separate features (X) and target (y) for both datasets
X_real = real_data.drop(columns=['Class'])
y_real = real_data['Class']
X_synthetic = synthetic_data

# Step 1: Compare Summary Statistics
print("\nSummary statistics for real dataset:")
print(real_data.describe())
print("\nSummary statistics for synthetic dataset:")
print(synthetic_data.describe())

# Step 2: Plot Feature Distributions
features = X_real.columns
for feature in features:
    plt.figure()
    plt.hist(X_real[feature], bins=20, alpha=0.5, label='Real Data')
    plt.hist(X_synthetic[feature], bins=20, alpha=0.5, label='Synthetic Data')
    plt.title(f'Distribution of {feature}')
    plt.legend()
    plt.show()

# Step 3: Assess Machine Learning Utility
# Adjust synthetic dataset to match the size of real training set
X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
    X_real, y_real, test_size=0.2, random_state=42)

# Ensure the synthetic data size matches the training size of the real data
X_synthetic = X_synthetic.sample(len(X_real_train), random_state=42, replace=True)

# Train on Synthetic, Test on Real (TSTR)
rf_tstr = RandomForestClassifier(random_state=42)
rf_tstr.fit(X_synthetic, y_real_train)  # Train on synthetic
y_pred_tstr = rf_tstr.predict(X_real_test)  # Test on real
accuracy_tstr = accuracy_score(y_real_test, y_pred_tstr)
print(f"\nTSTR Accuracy (Synthetic -> Real): {accuracy_tstr:.4f}")

# Train on Real, Test on Real (TRTR)
rf_trtr = RandomForestClassifier(random_state=42)
rf_trtr.fit(X_real_train, y_real_train)  # Train on real
y_pred_trtr = rf_trtr.predict(X_real_test)  # Test on real
accuracy_trtr = accuracy_score(y_real_test, y_pred_trtr)
print(f"TRTR Accuracy (Real -> Real): {accuracy_trtr:.4f}")

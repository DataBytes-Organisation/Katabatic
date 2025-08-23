# Mock CTGAN Training Script on Bank Personal Loan dataset
# Author: Preet Sureshbhai Jasoliya (CTGAN-preet branch)

import pandas as pd
import random

# Load dataset
data = pd.read_csv("data/Bank_Personal_Loan.csv").head(50)  # small sample for speed
if 'ID' in data.columns:
    data = data.drop(columns=['ID'])

print("Data loaded. Shape:", data.shape)

# Pretend CTGAN training
print("Starting CTGAN training (mock run)...")
for epoch in range(1, 4):  # fake 3 epochs
    print(f"Epoch {epoch}/3 - loss_G: {round(random.uniform(0.1, 1.0), 4)}, loss_D: {round(random.uniform(0.1, 1.0), 4)}")

print("Model training complete!")

# Generate synthetic samples (mocked)
synthetic = data.sample(5, replace=True).reset_index(drop=True)
synthetic["Age"] = synthetic["Age"] + random.randint(-2, 2)  # slight variation

print("Synthetic data sample:")
print(synthetic.head())

# Save synthetic dataset
synthetic.to_csv("models/ctgan-preet/synthetic_bankloan.csv", index=False)
print("Synthetic dataset saved at models/ctgan-preet/synthetic_bankloan.csv")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from katabatic.models.meg_DGEK.utils import get_demo_data
from katabatic.models.meg_DGEK.meg_adapter import MegAdapter

# Load and Display Sample Data
print("Loading demo data...")
df = get_demo_data('adult-raw')
print("Sample Data:")
print(df.head())

# Prepare Training and Testing Data
print("Preparing data...")
X, y = df.values[:, :-1], df.values[:, -1]  # Separate features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize and Train the MEG Model
print("Initializing and training the MEG model...")
adapter = MegAdapter()
adapter.load_model()
adapter.fit(X_train, y_train, epochs=5)  # Train for 5 epochs

# Generate Synthetic Data
print("Generating synthetic data...")
synthetic_data = adapter.generate(size=5)

# Display Generated Data
print("Synthetic Data:")
print(synthetic_data)

# Save Synthetic Data to CSV
print("Saving synthetic data to synthetic_data.csv...")
synthetic_data.to_csv('synthetic_data.csv', index=False)
print("Synthetic data saved successfully!")

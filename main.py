import torch
from goggle import GOGGLE
from goggle.datasets import data_factory

# Load a small built-in dataset (e.g., synthetic adult dataset)
dataset = data_factory(name="adult", data_dir="./data")

# Model parameters (small for testing)
model = GOGGLE(
    data=dataset.data,
    num_epochs=2,       # just to test run
    batch_size=64
)

# Train the model
model.train()

# Generate synthetic data
synthetic_data = model.generate(num_samples=5)
print("Generated Synthetic Data:")
print(synthetic_data)

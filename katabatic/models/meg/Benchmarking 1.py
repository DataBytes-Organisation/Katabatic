import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv('dataset.csv')

# Drop missing values (if any)
data = data.dropna()

# Convert categorical columns to numerical values
data['Gender'] = data['Gender'].map({'M': 1, 'F': 0})
data['Purchase'] = data['Purchase'].map({'Yes': 1, 'No': 0})

# Separate features (X) and labels (y)
X = data.drop('Purchase', axis=1)
y = data['Purchase']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Define the MaskedGenerator class
class MaskedGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, mask):
        super(MaskedGenerator, self).__init__()
        self.mask = mask
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x) * self.mask
        return output

# Define the Discriminator class
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Example usage:
input_dim = 100  # Noise dimension
output_dim = X_train_tensor.size(1)  # Output dimension (number of features in the data)

# Define masks for each generator
masks = [
    torch.tensor([1, 0, 0], dtype=torch.float32),  # Generator 1
    torch.tensor([0, 1, 0], dtype=torch.float32),  # Generator 2
    torch.tensor([0, 0, 1], dtype=torch.float32),  # Generator 3
]

# Ensure the masks sum to cover all output dimensions
assert sum(mask.sum() for mask in masks) == output_dim

# Initialize generators and the discriminator
generators = [MaskedGenerator(input_dim, output_dim, mask) for mask in masks]
discriminator = Discriminator(input_dim=output_dim)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)
optimizers_G = [optim.Adam(generator.parameters(), lr=0.0002) for generator in generators]

# Training loop
num_epochs = 5000
batch_size = 64
d_losses = []
g_losses = []

for epoch in range(num_epochs):
    # Train Discriminator
    discriminator.train()
    optimizer_D.zero_grad()

    # Real data
    idx = torch.randperm(X_train_tensor.size(0))[:batch_size]
    real_data = X_train_tensor[idx]
    real_labels = torch.ones((real_data.size(0), 1))

    # Fake data from ensemble of generators
    noise = torch.randn(real_data.size(0), input_dim)
    fake_data = sum(generator(noise) for generator in generators)
    fake_labels = torch.zeros((real_data.size(0), 1))

    # Compute discriminator loss
    real_output = discriminator(real_data)
    fake_output = discriminator(fake_data.detach())
    d_loss_real = criterion(real_output, real_labels)
    d_loss_fake = criterion(fake_output, fake_labels)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    optimizer_D.step()

    # Train each Generator
    for i, generator in enumerate(generators):
        optimizers_G[i].zero_grad()

        # Generate fake data
        noise = torch.randn(batch_size, input_dim)
        fake_data = generator(noise)

        # Compute generator loss
        fake_output = discriminator(fake_data)
        g_loss = criterion(fake_output, torch.ones((batch_size, 1)))
        g_loss.backward()
        optimizers_G[i].step()

    if epoch % 500 == 0:
        print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())

# Plot training loss curves for Discriminator and Generators
plt.figure(figsize=(10,5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Trend During Training')
plt.show()

# Generate synthetic data
def generate_synthetic_data(generators, num_samples):
    synthetic_parts = [generator(torch.randn(num_samples, input_dim)).detach().numpy() for generator in generators]
    combined_data = np.sum(synthetic_parts, axis=0)
    return combined_data

# Generate and save synthetic data
synthetic_data = generate_synthetic_data(generators, len(X_test))
synthetic_df = pd.DataFrame(synthetic_data, columns=X.columns)  # Use your original column names

# Plot real vs synthetic data for each feature
for col in X.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(X_test[col], color='blue', label='Real', kde=True, stat="density", linewidth=0)
    sns.histplot(synthetic_df[col], color='orange', label='Synthetic', kde=True, stat="density", linewidth=0)
    plt.title(f'Distribution of {col}: Real vs Synthetic')
    plt.legend()
    plt.show()

# Benchmarking: Test the generated synthetic data using classification metrics

# Re-scale synthetic data
synthetic_df_scaled = scaler.transform(synthetic_df)

# Convert to tensor
synthetic_tensor = torch.tensor(synthetic_df_scaled, dtype=torch.float32)

# Compare synthetic data with real data using classification metrics
def benchmark_synthetic_data(real_data, synthetic_data, real_labels):
    real_predictions = discriminator(real_data).detach().numpy()
    synthetic_predictions = discriminator(synthetic_data).detach().numpy()

    real_labels = real_labels.numpy()

    # Use threshold 0.5 for binary classification
    real_preds_class = (real_predictions > 0.5).astype(int)
    synthetic_preds_class = (synthetic_predictions > 0.5).astype(int)

    accuracy = accuracy_score(real_labels, real_preds_class)
    precision = precision_score(real_labels, real_preds_class)
    recall = recall_score(real_labels, real_preds_class)
    f1 = f1_score(real_labels, real_preds_class)

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Run benchmarking
benchmark_synthetic_data(X_test_tensor, synthetic_tensor, y_test_tensor)

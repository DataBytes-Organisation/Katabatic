from meg_adapter import MEG_Adapter

# Parameters
input_dim = 100  # Example noise vector dimension
output_dim = len(data.columns)  # Number of features in the dataset

# Initialize the MEG model
meg_adapter = MEG_Adapter(input_dim, output_dim)

# Train the model
meg_adapter.train(data_scaled, num_epochs=5000, batch_size=64)

# Generate Synthetic Data after training
synthetic_data = meg_adapter.generate(num_samples=1000)
print(synthetic_data)

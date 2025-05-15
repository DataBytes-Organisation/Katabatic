import torch
import pandas as pd
from meg_adapter import MEG_Adapter
from load_data import data_scaled  # Assuming you have the preprocessed data

# Parameters
input_dim = 100  # Example noise vector dimension
output_dim = len(data_scaled.columns)  # Number of features in the dataset

# Initialize the MEG model (Ensure the model is already trained)
meg_adapter = MEG_Adapter(input_dim, output_dim)

# Load the pretrained model (if you saved it previously)
# torch.load('path_to_saved_model.pth')  # Uncomment this line if you're using a saved model

# Generate Synthetic Data
num_samples = 1000  # Define the number of synthetic samples you want to generate
synthetic_data = meg_adapter.generate(num_samples)

# Convert synthetic data to a pandas DataFrame
synthetic_data_df = pd.DataFrame(synthetic_data.detach().numpy(), columns=data_scaled.columns)

# Save the synthetic data to a CSV file
synthetic_data_df.to_csv('synthetic_data.csv', index=False)

# Print the generated synthetic data
print("Generated synthetic data:")
print(synthetic_data_df.head())

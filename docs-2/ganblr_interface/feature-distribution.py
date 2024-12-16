import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Parse output directory argument
parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, help="Path to the folder to save output images")
args = parser.parse_args()
output_dir = args.output

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load datasets
real_data = pd.read_csv('credit_X_train.csv')
synthetic_data = pd.read_csv('credit_synthetic_data.csv')

# Ensure column names match
if list(real_data.columns) != list(synthetic_data.columns):
    synthetic_data.columns = real_data.columns

# Select key features for visualization
features = ['V1', 'V2', 'Amount', 'Time']

# Plot distributions and save
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(real_data[feature], label='Real Data', fill=True, alpha=0.5)
    sns.kdeplot(synthetic_data[feature], label='Synthetic Data', fill=True, alpha=0.5)
    plt.title(f'{feature} Distribution: Real vs Synthetic')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    
    # Save the image to the specified output folder
    output_path = os.path.join(output_dir, f"{feature}_comparison.png")
    plt.savefig(output_path)
    plt.close()

print(f"Feature distributions saved successfully in {output_dir}.")

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Argument parser
parser = argparse.ArgumentParser(description="Generate feature distribution plots.")
parser.add_argument("--real", required=True, help="Path to the real dataset file.")
parser.add_argument("--synthetic", required=True, help="Path to the synthetic dataset file.")
parser.add_argument("--output", required=True, help="Directory to save the output plots.")
args = parser.parse_args()

# Load datasets
real_data = pd.read_csv(args.real)
synthetic_data = pd.read_csv(args.synthetic)

# Ensure column names match
if list(real_data.columns) != list(synthetic_data.columns):
    synthetic_data.columns = real_data.columns

# Select key features for visualization
features = ['Amount', 'Time', 'V1', 'V2']  # Update this list based on available features

# Generate plots
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(real_data[feature], label='Real Data', fill=True, alpha=0.5)
    sns.kdeplot(synthetic_data[feature], label='Synthetic Data', fill=True, alpha=0.5)
    plt.title(f"{feature} Distribution: Real vs Synthetic")
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    output_path = os.path.join(args.output, f"{feature}_comparison.png")
    plt.savefig(output_path)
    plt.close()

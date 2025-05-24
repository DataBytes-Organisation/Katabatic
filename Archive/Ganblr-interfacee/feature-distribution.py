import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
real_data = pd.read_csv('credit_X_train.csv')
synthetic_data = pd.read_csv('credit_synthetic_data.csv')

# Ensure column names match
if list(real_data.columns) != list(synthetic_data.columns):
    synthetic_data.columns = real_data.columns

# Select key features for visualization
features = ['V1', 'V2', 'Amount', 'Time']

# Plot distributions
for feature in features:
    plt.figure(figsize=(8, 5))
    sns.kdeplot(real_data[feature], label='Real Data', fill=True, alpha=0.5)
    sns.kdeplot(synthetic_data[feature], label='Synthetic Data', fill=True, alpha=0.5)
    plt.title(f'{feature} Distribution: Real vs Synthetic')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'{feature}_comparison.png')
    plt.show()

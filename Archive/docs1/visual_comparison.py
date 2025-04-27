import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
real_data = pd.read_csv("car_dataset_encoded.csv")  # Real dataset
synthetic_data = pd.read_csv("car_synthetic_data.csv")  # Synthetic dataset

# Define features to compare
features = ["Buying", "Maint", "Doors", "Persons", "Lug_boot", "Safety"]

# Create histograms for feature distributions
def plot_histograms(real_data, synthetic_data, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(real_data[feature], kde=True, label="Real Data", color="blue", bins=20)
        sns.histplot(synthetic_data[feature], kde=True, label="Synthetic Data", color="orange", bins=20)
        plt.title(f"Feature Distribution: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

# Create boxplots for feature distributions
def plot_boxplots(real_data, synthetic_data, features):
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=[real_data[feature], synthetic_data[feature]], palette="Set2")
        plt.xticks([0, 1], ["Real Data", "Synthetic Data"])
        plt.title(f"Boxplot Comparison: {feature}")
        plt.ylabel(feature)
        plt.show()

# Generate histograms
print("Generating histograms for visual comparison...")
plot_histograms(real_data, synthetic_data, features)

# Generate boxplots
print("Generating boxplots for visual comparison...")
plot_boxplots(real_data, synthetic_data, features)

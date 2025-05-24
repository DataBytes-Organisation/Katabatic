import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import logging

# Add the project root to the Python path to allow module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

# Import CtganAdapter and evaluation functions from Katabatic
from katabatic.models.ctgan.ctgan_adapter import CtganAdapter
from katabatic.models.ctgan.ctgan_benchmark import evaluate_ctgan, print_evaluation_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_data_summary(data, title):
    """
    Print summary statistics of the dataset.
    
    Args:
        data (pd.DataFrame): The dataset to summarize.
        title (str): Title for the summary.
    """
    print(f"\n{title} Summary:")
    print(f"Shape: {data.shape}")
    print("\nDataset Head:")
    print(data.head())
    print("\nDataset Info:")
    data.info()
    print("\nNumeric Columns Summary:")
    print(data.describe())
    print("\nCategory Distribution:")
    print(data['Category'].value_counts(normalize=True))

def main():
    try:
        logger.info("Starting CT-GAN Iris example script")
        print("Starting CT-GAN Iris example script")

        # Load Iris dataset
        iris = load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target, name="Category").astype('category')
        data = pd.concat([X, y], axis=1)

        logger.info("Iris data loaded successfully")
        print_data_summary(data, "Original Iris Dataset")

        # Initialize CtganAdapter
        ctgan_params = {
            "noise_dim": 128,
            "learning_rate": 2e-4,
            "batch_size": 100,
            "discriminator_steps": 2,
            "epochs": 10,
            "lambda_gp": 10,
            "pac": 10,
            "cuda": False,
            "vgm_components": 2
        }
        ctgan_model = CtganAdapter(**ctgan_params)
        logger.info("CT-GAN model initialized with parameters")

        # Fit the CT-GAN model
        ctgan_model.fit(X, y)
        logger.info("CT-GAN model fitted successfully")

        # Generate synthetic data
        synthetic_data = ctgan_model.generate(n=len(data))
        synthetic_data = synthetic_data[data.columns]
        synthetic_data['Category'] = synthetic_data['Category'].astype('category')
        logger.info(f"Generated {len(synthetic_data)} rows of synthetic data")

        print_data_summary(synthetic_data, "Synthetic Iris Dataset")

        # Save synthetic data to CSV
        synthetic_data.to_csv("synthetic_iris_data.csv", index=False)
        logger.info("Synthetic Iris data saved to 'synthetic_iris_data.csv'")

        # Evaluate the quality of the synthetic data
        logger.info("Evaluating synthetic data quality")
        evaluation_metrics = evaluate_ctgan(real_data=data, synthetic_data=synthetic_data)
        
        print("\nEvaluation Metrics:")
        print_evaluation_results(evaluation_metrics)

        # Compare real and synthetic data distributions
        print("\nFeature Distribution Comparison:")
        for column in data.columns:
            if data[column].dtype != 'category':
                real_mean = data[column].mean()
                real_std = data[column].std()
                synth_mean = synthetic_data[column].mean()
                synth_std = synthetic_data[column].std()
                print(f"\n{column}:")
                print(f"  Real   - Mean: {real_mean:.4f}, Std: {real_std:.4f}")
                print(f"  Synth  - Mean: {synth_mean:.4f}, Std: {synth_std:.4f}")
            else:
                real_dist = data[column].value_counts(normalize=True)
                synth_dist = synthetic_data[column].value_counts(normalize=True)
                print(f"\n{column} Distribution:")
                print("  Real:")
                print(real_dist)
                print("  Synthetic:")
                print(synth_dist)

        logger.info("CT-GAN Iris example completed successfully")
        print("\nCT-GAN Iris example completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
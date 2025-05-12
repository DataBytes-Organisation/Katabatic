from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
import numpy as np
import pandas as pd
import glob
import time
import os

# Setup parameters based on the paper, but drastically reduced for quick demo
dataset = "Adult"
real_path = "Real_Datasets/Adult.csv"
fake_file_root = "Fake_Datasets"
num_exp = 1  # Reduced from 5 to 1 for quicker demo

# Create a custom CTABGAN class with much fewer epochs
class CTABGAN_Quick(CTABGAN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override the synthesizer's epochs setting
        self.synthesizer.epochs = 3  # Extremely reduced from 150 to 3 for very quick demo
        self.synthesizer.batch_size = 100  # Smaller batch size for faster processing

# Model parameters as used in the paper
synthesizer = CTABGAN_Quick(
    raw_csv_path=real_path,
    test_ratio=0.20,  # Test ratio as specified in the paper
    categorical_columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
    log_columns=[],
    mixed_columns={'capital-loss': [0.0], 'capital-gain': [0.0]},
    general_columns=["age"],
    non_categorical_columns=[],
    integer_columns=['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week'],
    problem_type={"Classification": 'income'}
)

# Create directory if it doesn't exist
os.makedirs(f"{fake_file_root}/{dataset}", exist_ok=True)

print(f"Starting CTAB-GAN+ (very quick demo) training for {dataset} dataset")
print(f"Number of experiments: {num_exp}")
print(f"Note: Using extremely few epochs (3 instead of 150) for quick demonstration")
print(f"Warning: Results will be of poor quality due to insufficient training")

# Start time for measuring total runtime
total_start_time = time.time()

# Run the experiments
for i in range(num_exp):
    print(f"\nExperiment {i+1}/{num_exp}:")
    start_time = time.time()
    
    print("Training model...")
    synthesizer.fit()
    
    print("Generating synthetic samples...")
    syn = synthesizer.generate_samples()
    
    # Save the synthetic data
    output_path = f"{fake_file_root}/{dataset}/{dataset}_fake_ultraquick_{i}.csv"
    syn.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")
    
    end_time = time.time()
    print(f"Experiment {i+1} completed in {end_time - start_time:.2f} seconds")

total_end_time = time.time()
print(f"\nAll experiments completed in {total_end_time - total_start_time:.2f} seconds")

# Get paths to all generated fake datasets
fake_paths = glob.glob(f"{fake_file_root}/{dataset}/*ultraquick*")
print(f"\nEvaluating {len(fake_paths)} synthetic datasets")

# Show sample of generated data
print("\nSample of generated data:")
sample_data = pd.read_csv(fake_paths[0])
print(sample_data.head(5))

print("\n---------------------------------------------------------")
print("CTAB-GAN+ Ultra-Quick Demo Results")
print("---------------------------------------------------------")
print("Note: These results are only for demonstration purposes and")
print("will be poor quality due to extremely limited training (3 epochs).")
print("For proper benchmarking matching the paper results, use:")
print("- 150 epochs (vs 3 used here)")
print("- 5 experiment runs (vs 1 used here)")
print("- Original batch size (500 vs 100 used here)")
print("---------------------------------------------------------") 
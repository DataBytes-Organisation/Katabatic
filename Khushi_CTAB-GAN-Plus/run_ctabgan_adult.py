from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
import numpy as np
import pandas as pd
import glob
import time
import os

# Setup parameters based on the paper
dataset = "Adult"
real_path = "Real_Datasets/Adult.csv"
fake_file_root = "Fake_Datasets"
num_exp = 5  # Number of experiments to run

# Model parameters as used in the paper
synthesizer = CTABGAN(
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

print(f"Starting CTAB-GAN+ training for {dataset} dataset")
print(f"Number of experiments: {num_exp}")

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
    output_path = f"{fake_file_root}/{dataset}/{dataset}_fake_{i}.csv"
    syn.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")
    
    end_time = time.time()
    print(f"Experiment {i+1} completed in {end_time - start_time:.2f} seconds")

total_end_time = time.time()
print(f"\nAll experiments completed in {total_end_time - total_start_time:.2f} seconds")

# Get paths to all generated fake datasets
fake_paths = glob.glob(f"{fake_file_root}/{dataset}/*")
print(f"\nEvaluating {len(fake_paths)} synthetic datasets")

# Perform evaluation using utility metrics (as specified in the paper)
print("\nEvaluating machine learning utility...")
model_dict = {"Classification": ["lr", "dt", "rf", "mlp", "svm"]}
result_mat = get_utility_metrics(real_path, fake_paths, "MinMax", model_dict, test_ratio=0.20)
result_df = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
result_df.index = list(model_dict.values())[0]
print("\nML Utility Metrics:")
print(result_df)

# Evaluate statistical similarity as in the paper
print("\nEvaluating statistical similarity...")
adult_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
stat_res_avg = []
for fake_path in fake_paths:
    stat_res = stat_sim(real_path, fake_path, adult_categorical)
    stat_res_avg.append(stat_res)

stat_columns = ["Average WD (Continuous Columns)", "Average JSD (Categorical Columns)", "Correlation Distance"]
stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)
print("\nStatistical Similarity Metrics:")
print(stat_results)

# Evaluate privacy metrics as in the paper
print("\nEvaluating privacy metrics...")
priv_res_avg = []
for fake_path in fake_paths:
    priv_res = privacy_metrics(real_path, fake_path)
    priv_res_avg.append(priv_res)

privacy_columns = [
    "DCR between Real and Fake (5th perc)",
    "DCR within Real (5th perc)",
    "DCR within Fake (5th perc)",
    "NNDR between Real and Fake (5th perc)",
    "NNDR within Real (5th perc)",
    "NNDR within Fake (5th perc)"
]
privacy_results = pd.DataFrame(np.array(priv_res_avg).mean(axis=0).reshape(1, 6), columns=privacy_columns)
print("\nPrivacy Metrics:")
print(privacy_results)

# Compare with results reported in the paper
print("\n---------------------------------------------------------")
print("CTAB-GAN+ Benchmark Results")
print("---------------------------------------------------------")
print("These metrics should be compared with those reported in the paper.")
print("Note: Small variations are expected due to randomness in training.")
print("---------------------------------------------------------") 
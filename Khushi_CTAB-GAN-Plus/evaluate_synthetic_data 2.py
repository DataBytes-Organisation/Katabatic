from model.eval.evaluation import get_utility_metrics, stat_sim, privacy_metrics
import numpy as np
import pandas as pd
import glob
import os

print("CTAB-GAN+ Synthetic Data Evaluation")
print("-----------------------------------")

# Set paths
dataset = "Adult"
real_path = "Real_Datasets/Adult.csv"
fake_file_root = "Fake_Datasets"
fake_paths = glob.glob(f"{fake_file_root}/{dataset}/*ultraquick*")

if not fake_paths:
    print("No synthetic datasets found. Please run the generation script first.")
    exit(1)

print(f"Found {len(fake_paths)} synthetic datasets to evaluate")
for path in fake_paths:
    print(f"- {path}")

# Perform evaluation using utility metrics (as specified in the paper)
print("\n1. Evaluating machine learning utility...")
model_dict = {"Classification": ["lr", "dt", "rf", "mlp", "svm"]}
result_mat = get_utility_metrics(real_path, fake_paths, "MinMax", model_dict, test_ratio=0.20)
result_df = pd.DataFrame(result_mat, columns=["Acc", "AUC", "F1_Score"])
result_df.index = list(model_dict.values())[0]
print("\nML Utility Metrics (lower values indicate better performance):")
print(result_df)

# Evaluate statistical similarity as in the paper
print("\n2. Evaluating statistical similarity...")
adult_categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
stat_res_avg = []
for fake_path in fake_paths:
    print(f"\nEvaluating {os.path.basename(fake_path)}:")
    stat_res = stat_sim(real_path, fake_path, adult_categorical)
    stat_res_avg.append(stat_res)

stat_columns = ["Average WD (Continuous Columns)", "Average JSD (Categorical Columns)", "Correlation Distance"]
stat_results = pd.DataFrame(np.array(stat_res_avg).mean(axis=0).reshape(1, 3), columns=stat_columns)
print("\nStatistical Similarity Metrics (lower values indicate better similarity):")
print(stat_results)

# Evaluate privacy metrics as in the paper
print("\n3. Evaluating privacy metrics...")
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
print("\nFor privacy, DCR (Distance to Closest Record) values between real and fake data")
print("should be similar to within-real and within-fake values to indicate good privacy.")

# Compare with results reported in the paper
print("\n---------------------------------------------------------")
print("CTAB-GAN+ Evaluation Results vs Paper")
print("---------------------------------------------------------")
print("Note: Our results are expected to be worse than those in the paper")
print("due to the extremely limited training (3 epochs vs 150 in the paper).")
print("This is only a demonstration of the evaluation process.")
print("\nPaper reported approximately:")
print("- ML Utility: Small differences (<5%) between real and synthetic data")
print("- Statistical Similarity: Low WD and JSD scores (<0.2)")
print("- Privacy: Comparable DCR values between real-fake and within distributions")
print("---------------------------------------------------------") 
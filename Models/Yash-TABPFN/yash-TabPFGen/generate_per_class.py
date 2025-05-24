import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ✅ Ensure TabPFGen module is importable
sys.path.append(os.path.join(os.getcwd(), "src", "tabpfngen_backup"))
from tabpfgen import TabPFGen  # importing from tabpfgen.py file

# Folder paths
DATASET_DIR = "datasets"
OUTPUT_DIR = "generated_per_class"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop over all CSV datasets
for filename in os.listdir(DATASET_DIR):
    if not filename.endswith(".csv"):
        continue

    dataset_path = os.path.join(DATASET_DIR, filename)
    print(f"\n🔍 Generating per-class data for: {filename}")

    try:
        df = pd.read_csv(dataset_path)

        # Assume target is last column
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Encode target if needed
        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        # Skip datasets with too few samples per class
        unique, counts = np.unique(y, return_counts=True)
        if np.min(counts) < 2:
            print(f"⚠️ Skipped {filename}: At least one class has <2 samples")
            continue

        # Train TabPFGen once per class
        synthetic_data = []
        for class_label in np.unique(y):
            # Filter samples of this class
            X_class = X[y == class_label]
            y_class = y[y == class_label]

            # Train-test split (80/20 for per-class)
            X_train, X_test, y_train, y_test = train_test_split(
                X_class, y_class, test_size=0.2, random_state=42
            )

            # Convert to numpy arrays
            X_train_np = X_train.to_numpy()
            y_train_np = y_train.to_numpy()

            # Train TabPFGen and generate samples
            generator = TabPFGen(n_sgld_steps=500)
            X_synth_np, y_synth_np = generator.generate_classification(
                X_train_np, y_train_np, n_samples=len(X_train_np)
            )

            df_synth = pd.DataFrame(X_synth_np, columns=X.columns)
            df_synth["target"] = class_label  # Inject correct label

            synthetic_data.append(df_synth)

        # Combine all per-class data and save
        final_df = pd.concat(synthetic_data, ignore_index=True)
        out_path = os.path.join(OUTPUT_DIR, f"{filename}_synth.csv")
        final_df.to_csv(out_path, index=False)
        print(f"✅ Saved synthetic data: {out_path}")

    except Exception as e:
        print(f"❌ Failed for {filename}: {e}")

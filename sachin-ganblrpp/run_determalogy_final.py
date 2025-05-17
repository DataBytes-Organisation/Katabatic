import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from katabatic.models.ganblrpp.ganblrpp_adapter import GanblrppAdapter

# ========================
# STEP 1: Load the Dataset
# ========================
print("\n🔹 Loading dermatology1 dataset...")
file_path = "katabatic/models/dermatology1.csv"
df = pd.read_csv(file_path)

# ========================
# STEP 2: Data Preparation
# ========================
# Remove 'id' column if it exists
if 'id' in df.columns:
    df = df.drop(columns=['id'])

# Separate features and target
target_col = "class"
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode categorical features
encoder_X = OrdinalEncoder()
X_encoded = encoder_X.fit_transform(X)

encoder_y = OrdinalEncoder()
y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()

# Get all possible classes from the encoder
all_classes = set(range(len(encoder_y.categories_[0])))

# Save full training data for global search
X_full = pd.DataFrame(X_encoded, columns=X.columns)
y_full = pd.Series(y_encoded)

# ========================
# STEP 3: Initialize Classifiers
# ========================
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Classifier": MLPClassifier(max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", n_estimators=100)
}

results = []

# ========================
# Enhanced Class Injection Logic
# ========================
def inject_missing_classes(X_synth, y_synth, X_train, y_train, all_classes, X_full, y_full):
    """
    Injects missing classes into the synthetic dataset.
    If not found in local split, searches full dataset.
    If still not found, injects a representative sample.
    """
    real_classes = set(np.unique(y_train))
    synth_classes = set(np.unique(y_synth))
    missing_classes = list(all_classes - synth_classes)

    if missing_classes:
        for missing_class in missing_classes:
            mask = y_train == missing_class
            X_missing = X_train[mask]
            y_missing = y_train[mask]

            if len(X_missing) == 0:
                print(f"❌ Class {missing_class} not found in local split. Searching full dataset.")
                mask_full = y_full == missing_class
                X_missing = X_full[mask_full]
                y_missing = y_full[mask_full]
                
            if len(X_missing) == 0:
                print(f"❌ Class {missing_class} not found in the full dataset. Injecting dummy sample.")
                X_mean = pd.DataFrame([X_full.mean(axis=0)], columns=X_synth.columns)
                X_missing = X_mean
                y_missing = pd.Series([missing_class])

            X_synth = pd.concat([X_synth, pd.DataFrame(X_missing, columns=X_synth.columns)], ignore_index=True)
            y_synth = pd.concat([y_synth, pd.Series(y_missing)], ignore_index=True)

    return X_synth, y_synth

# ========================
# STEP 4: Multi-Run Cross Validation (3 Repeats, 2-Fold CV)
# ========================
n_repeats = 3

for repeat in range(n_repeats):
    print(f"\n===== REPEAT {repeat + 1} =====")
    kf = KFold(n_splits=2, shuffle=True, random_state=42 + repeat)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_encoded)):
        print(f"\n===== FOLD {fold + 1} =====")

        sample_size = int(0.5 * len(train_idx))
        sampled_train_idx = np.random.choice(train_idx, sample_size, replace=False)

        X_train, X_test = X_encoded[sampled_train_idx], X_encoded[test_idx]
        y_train, y_test = y_encoded[sampled_train_idx], y_encoded[test_idx]

        # ========================
        # STEP 5: Train GANBLR++
        # ========================
        adapter = GanblrppAdapter(numerical_columns=[])
        adapter.load_model()
        adapter.fit(X_train, y_train, k=2, epochs=100, batch_size=64)

        # ========================
        # STEP 6: Generate Synthetic Data
        # ========================
        synthetic_df = adapter.generate(size=len(X_train))
        X_synth = synthetic_df.iloc[:, :-1]
        y_synth_raw = synthetic_df.iloc[:, -1].astype(int)

        X_synth.columns = X.columns
        X_synth = X_synth.astype(float)
        y_synth = y_synth_raw.astype(int)

        # Inject missing classes if detected
        X_synth, y_synth = inject_missing_classes(X_synth, y_synth, X_train, y_train, all_classes, X_full, y_full)

        # ========================
        # STEP 7: TSTR Evaluation + JSD + Wasserstein
        # ========================
        fold_results = {"Repeat": repeat + 1, "Fold": fold + 1}
        jsd_scores = []
        wd_scores = []

        for name, model in classifiers.items():
            model.fit(X_synth, y_synth)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            fold_results[name] = acc
            jsd_scores.append(jensenshannon(y_test, y_pred))
            wd_scores.append(wasserstein_distance(y_test, y_pred))

        fold_results["JSD"] = np.mean(jsd_scores)
        fold_results["Wasserstein"] = np.mean(wd_scores)
        results.append(fold_results)

# ========================
# STEP 8: Calculate Overall Averages and Save
# ========================
print("\n🔹 Calculating Overall Averages...")
average_results = {}

# List of classifiers and metrics
metrics = ["Logistic Regression", "MLP Classifier", "Random Forest", "XGBoost", "JSD", "Wasserstein"]

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results)

# Compute the average for each metric across all folds and repeats
for metric in metrics:
    average_results[metric] = results_df[metric].mean()

# Create a DataFrame for better presentation
average_df = pd.DataFrame([average_results])

# Save directly to CSV (overwrite the previous file)
average_df.to_csv("determalogy_final_tstr_results.csv", index=False)

# Display the overall averages
print("\n✅ Overall Averages Calculated and Saved:")
print(average_df)


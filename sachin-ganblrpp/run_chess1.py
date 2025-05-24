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
print("\n🔹 Loading chess1 dataset...")
file_path = "katabatic/models/chess1.csv"
df = pd.read_csv(file_path)

# ========================
# STEP 2: Data Preparation
# ========================
# Remove 'id' column
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
# STEP 4: Multi-Run Cross Validation (3 Repeats, 2-Fold CV)
# ========================
n_repeats = 3

for repeat in range(n_repeats):
    print(f"\n===== REPEAT {repeat + 1} =====")
    kf = KFold(n_splits=2, shuffle=True, random_state=42 + repeat)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_encoded)):
        print(f"\n===== FOLD {fold + 1} =====")

        # 50% Sample Size of the training data
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

        # Fix synthetic data columns
        X_synth.columns = X.columns
        X_synth = X_synth.astype(float)
        y_synth = y_synth_raw.astype(int)

        # ========================
        # STEP 7: TSTR Evaluation + JSD + Wasserstein
        # ========================
        for name, model in classifiers.items():
            print(f"🔸 {name}")
            try:
                model.fit(X_synth, y_synth)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # Compute JSD and Wasserstein Distance
                jsd = jensenshannon(y_test, y_pred)
                wd = wasserstein_distance(y_test, y_pred)

                results.append({
                    "Repeat": repeat + 1,
                    "Fold": fold + 1,
                    "Classifier": name,
                    "Accuracy": acc,
                    "JSD": jsd,
                    "Wasserstein": wd
                })
                print(f"✅ Accuracy: {acc:.4f} | JSD: {jsd:.4f} | WD: {wd:.4f}")
            except Exception as e:
                print(f"❌ {name} failed: {e}")
                results.append({
                    "Repeat": repeat + 1,
                    "Fold": fold + 1,
                    "Classifier": name,
                    "Accuracy": "Error",
                    "JSD": "Error",
                    "Wasserstein": "Error"
                })

# ========================
# STEP 8: Save Results
# ========================
results_df = pd.DataFrame(results)
results_df.to_csv("chess1_tstr_results.csv", index=False)
print("\n✅ Results saved to chess1_tstr_results.csv")

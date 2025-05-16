import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
import sys

# Setup path to tabpfgen implementation
sys.path.insert(0, './src/tabpfngen_backup')
from tabpfgen import TabPFGen

print("🔹 Loading mfeat_morphological dataset...")
df = pd.read_csv("datasets/mfeat_morphological.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# 80/20 split
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("🔁 Training TabPFGen per class (300 epochs)...")
X_synth_list, y_synth_list = [], []

for class_label in sorted(y_train_real.unique()):
    X_class = X_train_real[y_train_real == class_label]
    y_class = y_train_real[y_train_real == class_label]

    try:
        generator = TabPFGen(n_sgld_steps=300)
        X_synth, _ = generator.generate_classification(
            X_class.to_numpy(), y_class.to_numpy(), n_samples=len(X_class)
        )
        X_synth_list.append(X_synth)
        y_synth_list.append(np.full(len(X_class), class_label))
    except Exception as e:
        print(f"❌ Failed for class {class_label}: {e}")

if X_synth_list:
    X_train_synth = np.vstack(X_synth_list)
    y_train_synth = np.hstack(y_synth_list)

    print("\n🔍 Evaluating Classifiers:")

    # Logistic Regression
    try:
        lr_real = LogisticRegression(max_iter=1000).fit(X_train_real, y_train_real)
        acc_real = accuracy_score(y_test_real, lr_real.predict(X_test_real))
        lr_synth = LogisticRegression(max_iter=1000).fit(X_train_synth, y_train_synth)
        acc_synth = accuracy_score(y_test_real, lr_synth.predict(X_test_real))
        print(f"✅ Logistic Regression  | Real: {acc_real:.4f}")
        print(f"✅ Logistic Regression  | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ Logistic Regression failed: {e}")

    # Random Forest
    try:
        rf_real = RandomForestClassifier().fit(X_train_real, y_train_real)
        acc_real = accuracy_score(y_test_real, rf_real.predict(X_test_real))
        rf_synth = RandomForestClassifier().fit(X_train_synth, y_train_synth)
        acc_synth = accuracy_score(y_test_real, rf_synth.predict(X_test_real))
        print(f"✅ Random Forest        | Real: {acc_real:.4f}")
        print(f"✅ Random Forest        | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")

    # XGBoost
    try:
        xgb_real = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        xgb_real.fit(X_train_real, y_train_real)
        acc_real = accuracy_score(y_test_real, xgb_real.predict(X_test_real))

        xgb_synth = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        xgb_synth.fit(X_train_synth, y_train_synth)
        acc_synth = accuracy_score(y_test_real, xgb_synth.predict(X_test_real))

        print(f"✅ XGBoost              | Real: {acc_real:.4f}")
        print(f"✅ XGBoost              | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")

    # TabPFN Classifier
    try:
        tabpfn_real = TabPFNClassifier(device="cpu")
        tabpfn_real.fit(X_train_real.values, y_train_real.values)
        acc_real = accuracy_score(y_test_real, tabpfn_real.predict(X_test_real.values))

        tabpfn_synth = TabPFNClassifier(device="cpu")
        tabpfn_synth.fit(X_train_synth, y_train_synth)
        acc_synth = accuracy_score(y_test_real, tabpfn_synth.predict(X_test_real.values))

        print(f"✅ TabPFN               | Real: {acc_real:.4f}")
        print(f"✅ TabPFN               | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ TabPFN failed: {e}")
else:
    print("❌ No synthetic data was generated. Evaluation skipped.")

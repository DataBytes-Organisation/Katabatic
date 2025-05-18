import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import sys

# Add path to TabPFGen
sys.path.insert(0, './src/tabpfngen_backup')
from tabpfgen import TabPFGen

print("🔹 Loading KR-vs-KP dataset...")

df = pd.read_csv("datasets/kr-vs-kp.csv")
df["class"] = df["class"].map({"won": 1, "nowin": 0})

X = df.drop(columns=["class"])
y = df["class"]

# Encode categorical features
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

print("🔁 Training TabPFGen per class (300 epochs)...")
X_synth_list, y_synth_list = [], []

for label in sorted(np.unique(y_train_real)):
    X_sub = X_train_real[y_train_real == label]
    y_sub = y_train_real[y_train_real == label]
    try:
        generator = TabPFGen(n_sgld_steps=300)
        X_synth, _ = generator.generate_classification(X_sub, y_sub.to_numpy(), n_samples=len(X_sub))
        X_synth_list.append(X_synth)
        y_synth_list.append(np.full(len(X_sub), label))
    except Exception as e:
        print(f"❌ Failed for class {label}: {e}")

if X_synth_list:
    X_train_synth = np.vstack(X_synth_list)
    y_train_synth = np.hstack(y_synth_list)

    print("\n🔍 Evaluating Classifiers:\n")

    try:
        lr = LogisticRegression(max_iter=1000).fit(X_train_real, y_train_real)
        acc_real = accuracy_score(y_test_real, lr.predict(X_test_real))
        lr_s = LogisticRegression(max_iter=1000).fit(X_train_synth, y_train_synth)
        acc_synth = accuracy_score(y_test_real, lr_s.predict(X_test_real))
        print(f"✅ Logistic Regression | Real: {acc_real:.4f} | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ Logistic Regression failed: {e}")

    try:
        rf = RandomForestClassifier().fit(X_train_real, y_train_real)
        acc_real = accuracy_score(y_test_real, rf.predict(X_test_real))
        rf_s = RandomForestClassifier().fit(X_train_synth, y_train_synth)
        acc_synth = accuracy_score(y_test_real, rf_s.predict(X_test_real))
        print(f"✅ Random Forest        | Real: {acc_real:.4f} | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")

    try:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train_real, y_train_real)
        acc_real = accuracy_score(y_test_real, xgb.predict(X_test_real))
        xgb_s = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train_synth, y_train_synth)
        acc_synth = accuracy_score(y_test_real, xgb_s.predict(X_test_real))
        print(f"✅ XGBoost              | Real: {acc_real:.4f} | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")
else:
    print("❌ No synthetic data was generated. Evaluation skipped.")

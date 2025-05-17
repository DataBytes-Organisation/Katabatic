import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ✅ Add custom TabPFGen path
import sys
sys.path.insert(0, './src/tabpfngen_backup')
from tabpfgen import TabPFGen

print("🔹 Loading Blood Transfusion dataset...")

df = pd.read_csv("datasets/blood-transfusion-service-center.csv")
df.columns = ["V1", "V2", "V3", "V4", "Class"]

X = df.drop(columns=["Class"])
y = df["Class"]

# 🔁 80/20 train-test split
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

    # ✅ Filter out synthetic samples with unexpected labels
    valid_classes = np.unique(y_test_real)
    mask = np.isin(y_train_synth, valid_classes)
    X_synth_filtered = X_train_synth[mask]
    y_synth_filtered = y_train_synth[mask]

    # ✅ Normalize labels using label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_test_real)

    y_test_real_enc = label_encoder.transform(y_test_real)
    y_train_real_enc = label_encoder.transform(y_train_real)
    y_synth_filtered_enc = label_encoder.transform(y_synth_filtered)

    print("\n🔍 Evaluating Classifiers:\n")

    # Logistic Regression
    try:
        lr_real = LogisticRegression(max_iter=1000).fit(X_train_real, y_train_real_enc)
        acc_real = accuracy_score(y_test_real_enc, lr_real.predict(X_test_real))
        lr_synth = LogisticRegression(max_iter=1000).fit(X_synth_filtered, y_synth_filtered_enc)
        acc_synth = accuracy_score(y_test_real_enc, lr_synth.predict(X_test_real))
        print(f"✅ Logistic Regression | Real: {acc_real:.4f} | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ Logistic Regression failed: {e}")

    # Random Forest
    try:
        rf_real = RandomForestClassifier().fit(X_train_real, y_train_real_enc)
        acc_real = accuracy_score(y_test_real_enc, rf_real.predict(X_test_real))
        rf_synth = RandomForestClassifier().fit(X_synth_filtered, y_synth_filtered_enc)
        acc_synth = accuracy_score(y_test_real_enc, rf_synth.predict(X_test_real))
        print(f"✅ Random Forest        | Real: {acc_real:.4f} | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ Random Forest failed: {e}")

    # XGBoost
    try:
        xgb_real = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train_real, y_train_real_enc)
        acc_real = accuracy_score(y_test_real_enc, xgb_real.predict(X_test_real))
        xgb_synth = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_synth_filtered, y_synth_filtered_enc)
        acc_synth = accuracy_score(y_test_real_enc, xgb_synth.predict(X_test_real))
        print(f"✅ XGBoost              | Real: {acc_real:.4f} | Synthetic: {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ XGBoost failed: {e}")
else:
    print("❌ No synthetic data was generated. Evaluation skipped.")

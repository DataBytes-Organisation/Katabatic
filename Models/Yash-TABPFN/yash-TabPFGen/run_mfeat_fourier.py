import sys
sys.path.append("src/tabpfngen_backup")
import tabpfgen


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
import sys
sys.path.append("src/tabpfngen_backup")
from tabpfgen import TabPFGen

print("🔹 Loading mfeat_fourier dataset...")

# Load and prepare data
df = pd.read_csv("datasets/mfeat_fourier.csv")
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

# Split real dataset
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Generate synthetic data using TabPFGen
print("🔁 Training TabPFGen per class (300 epochs)...")
X_synthetic_all = []
y_synthetic_all = []

for label in sorted(y.unique()):
    X_class = X_train_real[y_train_real == label].values
    y_class = y_train_real[y_train_real == label].values

    try:
        generator = TabPFGen(n_sgld_steps=300)
        X_syn, y_syn = generator.generate_classification(X_class, y_class, num_samples=len(X_class))
        X_synthetic_all.append(X_syn)
        y_synthetic_all.append(np.full(len(X_syn), label))
    except Exception as e:
        print(f"❌ Failed for class {label}: {e}")

if not X_synthetic_all:
    print("❌ No synthetic data was generated. Evaluation skipped.")
    exit()

X_synth = np.vstack(X_synthetic_all)
y_synth = np.concatenate(y_synthetic_all)

# Prepare test set
X_test = X_test_real
y_test = y_test_real

print("\n🔍 Evaluating Classifiers:")

# Helper function to evaluate and print
def evaluate_model(name, model, real=True):
    try:
        model.fit(X_train_real if real else X_synth, y_train_real if real else y_synth)
        acc = model.score(X_test, y_test)
        source = "Real" if real else "Synthetic"
        print(f"✅ {name:<20} | {source}: {acc:.4f}")
        return acc
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return None

# Logistic Regression
evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000), real=True)
evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000), real=False)

# Random Forest
evaluate_model("Random Forest", RandomForestClassifier(), real=True)
evaluate_model("Random Forest", RandomForestClassifier(), real=False)

# XGBoost
evaluate_model("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), real=True)
evaluate_model("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"), real=False)

# TabPFN
try:
    model_tabpfn = TabPFNClassifier(device="cpu")
    evaluate_model("TabPFN", model_tabpfn, real=True)
    evaluate_model("TabPFN", model_tabpfn, real=False)
except Exception as e:
    print(f"❌ TabPFN failed: {e}")

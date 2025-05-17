import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ✅ Manually add the backup folder to sys.path
import sys
sys.path.insert(0, './src/tabpfngen_backup') 
from tabpfgen import TabPFGen  # Now this works because tabpfgen.py is inside tabpfngen_backup/

# Load dataset
print("🔹 Loading Phoneme dataset...")
df = pd.read_csv(r"C:\Users\Manthan Goyal\Desktop\Team-Project\TabPFGen\datasets\phoneme.csv")
# Set up features and labels
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# 80/20 train/test split
X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train TabPFGen per class
print("🔁 Training TabPFGen per class (300 epochs)...")
generator = TabPFGen(n_sgld_steps=300)
X_synth_list, y_synth_list = [], []

for class_label in np.unique(y_train_real):
    class_mask = y_train_real == class_label
    X_class = X_train_real[class_mask]
    y_class = y_train_real[class_mask]

    try:
        X_gen, y_gen = generator.generate_classification(X_class.to_numpy(), y_class, n_samples=len(X_class))
        X_synth_list.append(X_gen)
        y_synth_list.append(np.full(len(X_gen), class_label))
    except Exception as e:
        print(f"❌ Failed for class {class_label}: {e}")

X_train_synth = np.vstack(X_synth_list)
y_train_synth = np.concatenate(y_synth_list)

# Classifiers
classifiers = {
    "LR": LogisticRegression(max_iter=500),
    "RF": RandomForestClassifier(),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric="logloss")
}

# Evaluation on Real Data
print("\n🎯 Accuracy on Real Data:")
for name, model in classifiers.items():
    model.fit(X_train_real, y_train_real)
    y_pred_real = model.predict(X_test_real)
    acc_real = accuracy_score(y_test_real, y_pred_real)
    print(f"✅ {name} Accuracy (Real): {acc_real:.4f}")

# Evaluation on Synthetic Data
print("\n🎯 Accuracy on Synthetic Data:")
for name, model in classifiers.items():
    try:
        model.fit(X_train_synth, y_train_synth)
        y_pred_synth = model.predict(X_test_real)
        acc_synth = accuracy_score(y_test_real, y_pred_synth)
        print(f"✅ {name} Accuracy (Synthetic): {acc_synth:.4f}")
    except Exception as e:
        print(f"❌ {name} failed on synthetic data: {e}")

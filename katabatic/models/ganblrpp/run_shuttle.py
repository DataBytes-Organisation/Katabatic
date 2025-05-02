import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from katabatic.models.ganblrpp.ganblrpp_adapter import GanblrppAdapter

# STEP 1: Load dataset
print("\n📥 Loading Shuttle dataset...")
df = pd.read_csv("katabatic/models/shuttle.csv")  # <- your uploaded shuttle file

# 🚀 Subsample if needed
if len(df) > 3000:
    df = df.sample(n=3000, random_state=42)
print(f"✅ Dataset shape after subsampling (if applied): {df.shape}")

# STEP 2: Select target column automatically
possible_targets = [col for col in df.columns if df[col].nunique() < 50 and df[col].nunique() > 1]
target_col = possible_targets[-1] if possible_targets else df.columns[-1]
print(f"🎯 Target column selected: '{target_col}'")

# STEP 3: Prepare X and y
X = df.drop(columns=[target_col]).fillna("Missing").astype(str)
y = df[target_col].fillna("Missing").astype(str)

encoder_X = OrdinalEncoder()
X_encoded = encoder_X.fit_transform(X)

encoder_y = OrdinalEncoder()
y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()

# STEP 4: 🚨 REMOVE classes with only 1 sample
value_counts = pd.Series(y_encoded).value_counts()
valid_classes = value_counts[value_counts > 1].index
mask = np.isin(y_encoded, valid_classes)
X_encoded = X_encoded[mask]
y_encoded = y_encoded[mask]
print(f"✅ Classes with >1 samples retained: {len(valid_classes)} classes")

# STEP 5: Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Classifier": MLPClassifier(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(eval_metric="mlogloss")
}

results = []

# STEP 6: Run 10 experiments
for run in range(1, 11):
    print(f"\n===== RUN {run} =====")
    seed = 2000 + run
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
    )

    adapter = GanblrppAdapter(numerical_columns=[])
    adapter.load_model()
    adapter.fit(X_train, y_train, k=2, epochs=10, batch_size=64)

    synthetic_df = adapter.generate(size=len(X_train))
    X_synth = synthetic_df.iloc[:, :-1]
    y_synth_raw = synthetic_df.iloc[:, -1].astype(int)

    # 🔥 Inject missing classes if needed
    real_classes = set(np.unique(y_train))
    synth_classes = set(np.unique(y_synth_raw))
    missing = list(real_classes - synth_classes)
    if missing:
        print(f"⚠️ Injecting missing class(es): {missing}")
        mask = np.isin(y_train, missing)
        X_synth = pd.concat([X_synth, pd.DataFrame(X_train[mask])], ignore_index=True)
        y_synth_raw = pd.concat([y_synth_raw, pd.Series(y_train[mask])], ignore_index=True)

    X_synth = X_synth.astype(float)
    X_test = pd.DataFrame(X_test).astype(float)
    y_synth

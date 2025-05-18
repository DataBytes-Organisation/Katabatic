import sys
import os
sys.path.append(os.path.abspath("."))

import pandas as pd
from katabatic.models.ganblrpp.ganblrpp_adapter import GanblrppAdapter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# STEP 1: Load dataset
print("\n[STEP 1] Loading dataset...")
df = pd.read_csv(r"C:\Users\sachi\OneDrive\Desktop\t1-2025\sit764 team project A\Katabatic-Git\Katabatic\katabatic\nursery.csv")
print(f"Dataset shape: {df.shape}")

X = df.drop("Target", axis=1)
y = df["Target"]

# STEP 2: Encode features and labels
X_encoded = pd.get_dummies(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# STEP 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
numerical_columns = X_encoded.select_dtypes(include=["int64", "float64"]).columns.tolist()
print(f"Numerical columns: {numerical_columns if numerical_columns else 'None'}")

# STEP 4: Train GANBLR++
print("\n[STEP 4] Training GANBLR++...")
adapter = GanblrppAdapter(numerical_columns=numerical_columns)
adapter.load_model()
adapter.fit(X_train, y_train, k=2, epochs=10, batch_size=64)
print("[SUCCESS] Model training completed")

# STEP 5: Generate synthetic data
print("\n[STEP 5] Generating synthetic samples...")
synthetic_df = adapter.generate(size=len(X_train))
print("[SUCCESS] Data generation completed")

X_synth = synthetic_df.iloc[:, :-1]
y_synth = synthetic_df.iloc[:, -1].astype(int)

# STEP 6: Ensure labels match between real and synthetic data
real_labels = set(y_test)
synth_labels = set(y_synth)
shared_labels = real_labels & synth_labels

if shared_labels != real_labels or shared_labels != synth_labels:
    print(f"[WARNING] Adjusting labels to shared set: {shared_labels}")
    y_test_series = pd.Series(y_test, index=X_test.index)
    y_synth_series = pd.Series(y_synth, index=X_synth.index)

    mask_real = y_test_series.isin(shared_labels)
    mask_synth = y_synth_series.isin(shared_labels)

    X_test = X_test.loc[mask_real]
    y_test = y_test_series.loc[mask_real].values

    X_synth = X_synth.loc[mask_synth]
    y_synth = y_synth_series.loc[mask_synth].values

if len(X_synth) == 0 or len(X_test) == 0:
    print("[ERROR] No valid samples remain after filtering. Aborting.")
    sys.exit(1)

# STEP 7: TSTR Evaluation
print("\n[STEP 7] TSTR Evaluation")
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Classifier": MLPClassifier(max_iter=300, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
}

results = {}
for name, model in classifiers.items():
    try:
        model.fit(X_synth, y_synth)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"[ERROR] {name} failed: {e}")

# STEP 8: Plot
plt.bar(results.keys(), results.values())
plt.title("TSTR Accuracy (Synthetic → Real)")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

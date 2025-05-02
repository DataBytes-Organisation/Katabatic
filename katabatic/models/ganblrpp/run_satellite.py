import sys, os
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

print("\n🔹 Loading Satellite dataset...")
df = pd.read_csv("katabatic/models/Satellite.csv")

# Select only relevant features
selected_features = [
    "Country of Operator/Owner", 
    "Operator/Owner", 
    "Purpose", 
    "Class of Orbit", 
    "Type of Orbit", 
    "Contractor", 
    "Country of Contractor", 
    "Launch Vehicle", 
    "Launch Site", 
    "Date of Launch"
]
target_col = "Launch Site"

# Drop missing target values
df = df[selected_features].dropna(subset=[target_col])
X = df.drop(columns=[target_col]).fillna("Missing").astype(str)
y = df[target_col].fillna("Missing").astype(str)

# Encode categorical features
encoder_X = OrdinalEncoder()
X_encoded = encoder_X.fit_transform(X)

encoder_y = OrdinalEncoder()
y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()

# Remove globally rare classes
value_counts = pd.Series(y_encoded).value_counts()
valid_classes = value_counts[value_counts > 1].index
mask = np.isin(y_encoded, valid_classes)
X_encoded = X_encoded[mask]
y_encoded = y_encoded[mask]

print(f"✅ Filtered dataset size: {len(X_encoded)}")

# Setup classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Classifier": MLPClassifier(max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}

results = []
split_size = len(X_encoded) // 10

for run in range(10):
    print(f"\n===== RUN {run + 1} =====")
    start, end = run * split_size, (run + 1) * split_size if run < 9 else len(X_encoded)

    X_part = X_encoded[start:end]
    y_part = y_encoded[start:end]

    # Check if split is valid
    unique, counts = np.unique(y_part, return_counts=True)
    if len(unique) < 2 or np.any(counts < 2):
        print("⚠️ Skipping RUN due to too few samples or classes.")
        continue

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X_part, y_part, test_size=0.2, random_state=42, stratify=y_part
    )

    # Train GANBLR++
    adapter = GanblrppAdapter(numerical_columns=[])
    adapter.load_model()
    adapter.fit(X_train, y_train, k=1, epochs=5, batch_size=8)

    # Generate synthetic data
    synthetic_df = adapter.generate(size=len(X_train))
    X_synth = synthetic_df.iloc[:, :-1].astype(float)
    y_synth_raw = synthetic_df.iloc[:, -1].astype(int)

    # Inject missing classes
    real_classes = set(np.unique(y_train))
    synth_classes = set(np.unique(y_synth_raw))
    missing = list(real_classes - synth_classes)
    if missing:
        print(f"⚠️ Injecting missing class(es): {missing}")
        mask_missing = np.isin(y_train, missing)
        X_missing = pd.DataFrame(X_train[mask_missing])
        y_missing = pd.Series(y_train[mask_missing])
        X_synth = pd.concat([X_synth, X_missing], ignore_index=True)
        y_synth_raw = pd.concat([y_synth_raw, y_missing], ignore_index=True)

    # Prepare data
    X_synth.columns = [str(i) for i in range(X_synth.shape[1])]
    X_test = pd.DataFrame(X_test, columns=X_synth.columns).astype(float)
    y_synth = y_synth_raw[:len(X_synth)].astype(int)

    # TSTR evaluation
    for name, model in classifiers.items():
        print(f"🔸 {name}")
        try:
            model.fit(X_synth, y_synth)
            acc = accuracy_score(y_test, model.predict(X_test))
            results.append({"Run": run + 1, "Classifier": name, "Accuracy": acc})
            print(f"✅ Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append({"Run": run + 1, "Classifier": name, "Accuracy": "Error"})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("satellite_tstr_results.csv", index=False)
print("\n✅ Results saved to satellite_tstr_results.csv")

# Plot
valid = results_df[results_df["Accuracy"] != "Error"]
valid["Accuracy"] = valid["Accuracy"].astype(float)
avg = valid.groupby("Classifier")["Accuracy"].mean()

plt.figure(figsize=(8, 5))
plt.bar(avg.index, avg.values)
plt.title("Average TSTR Accuracy (Satellite Dataset)")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("satellite_tstr_accuracy_summary.png")
plt.show()

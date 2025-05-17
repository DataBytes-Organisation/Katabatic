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
print("\n🔹 Loading MAGIC Gamma dataset...")
df = pd.read_csv("katabatic/models/magic.csv")
target_col = "class"

X = df.drop(columns=[target_col])
y = df[target_col]

numerical_cols = X.columns.tolist()
numerical_indices = list(range(len(numerical_cols)))

# STEP 2: Encode target only
encoder_y = OrdinalEncoder()
y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()
X_encoded = X.astype(float).values

# STEP 3: Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Classifier": MLPClassifier(max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}

results = []

# STEP 4: Run 10x 80/20 splits
for run in range(1, 11):
    print(f"\n===== RUN {run} =====")
    seed = 600 + run
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
    )

    # Train GANBLR++
    adapter = GanblrppAdapter(numerical_columns=numerical_indices)
    adapter.load_model()
    adapter.fit(X_train, y_train, k=2, epochs=300, batch_size=64)

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
        mask = np.isin(y_train, missing)
        X_synth = pd.concat([X_synth, pd.DataFrame(X_train[mask])], ignore_index=True)
        y_synth_raw = pd.concat([y_synth_raw, pd.Series(y_train[mask])], ignore_index=True)

    y_synth = y_synth_raw.astype(int)
    X_test = pd.DataFrame(X_test).astype(float)

    # TSTR Evaluation
    for name, model in classifiers.items():
        print(f"🔸 {name}")
        try:
            model.fit(X_synth, y_synth)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({"Run": run, "Classifier": name, "Accuracy": acc})
            print(f"✅ Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append({"Run": run, "Classifier": name, "Accuracy": "Error"})

# STEP 5: Save results
results_df = pd.DataFrame(results)
results_df.to_csv("magic_tstr_results.csv", index=False)
print("\n✅ Results saved to magic_tstr_results.csv")

# STEP 6: Plot average accuracy
valid = results_df[results_df["Accuracy"] != "Error"]
valid["Accuracy"] = valid["Accuracy"].astype(float)
avg = valid.groupby("Classifier")["Accuracy"].mean()

plt.figure(figsize=(8, 5))
plt.bar(avg.index, avg.values, color="darkorange")
plt.title("Average TSTR Accuracy over 10 Runs - MAGIC Gamma Dataset")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("magic_tstr_accuracy_summary.png")
plt.show()

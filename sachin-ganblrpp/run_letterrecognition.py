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

# Load the letterrecognition dataset
print("\n📥 Loading letterrecognition.csv...")
df = pd.read_csv("katabatic/models/letterrecognition.csv")

# Auto-select target (first column, because letter recognition datasets often have target first)
target_col = df.columns[0]
print(f"🎯 Target column selected: '{target_col}'")

# Separate features and target
X = df.drop(columns=[target_col]).fillna("Missing").astype(str)
y = df[target_col].fillna("Missing").astype(str)

# Ordinal encode
encoder_X = OrdinalEncoder()
X_encoded = encoder_X.fit_transform(X)

encoder_y = OrdinalEncoder()
y_encoded = encoder_y.fit_transform(y.values.reshape(-1, 1)).astype(int).ravel()

# Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "MLP Classifier": MLPClassifier(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}

results = []

# 10 different runs
for run in range(1, 11):
    print(f"\n===== RUN {run} =====")
    seed = 1000 + run
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
    )

    # Reduce to 2500 samples max for training (optional)
    if len(X_train) > 2500:
        X_train = X_train[:2500]
        y_train = y_train[:2500]

    for name, model in classifiers.items():
        print(f"🔸 {name}")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({"Run": run, "Classifier": name, "Accuracy": acc})
            print(f"✅ Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results.append({"Run": run, "Classifier": name, "Accuracy": "Error"})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("letterrecognition_tstr_results.csv", index=False)
print("\n✅ Results saved to letterrecognition_tstr_results.csv")

# Plot average accuracy
valid = results_df[results_df["Accuracy"] != "Error"]
valid["Accuracy"] = valid["Accuracy"].astype(float)
avg = valid.groupby("Classifier")["Accuracy"].mean()

plt.figure(figsize=(8, 5))
plt.bar(avg.index, avg.values)
plt.title("Average TSTR Accuracy over 10 Runs - Letter Recognition Dataset")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("letterrecognition_tstr_accuracy_summary.png")
plt.show()

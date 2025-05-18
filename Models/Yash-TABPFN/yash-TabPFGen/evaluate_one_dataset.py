import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 📌 CHANGE THIS TO YOUR FILE
filename = "generated_per_class/mfeat-factors.csv_synth.csv"

# 🔹 Load the synthetic dataset
df = pd.read_csv(filename)
target_col = df.columns[-1]  # assume last column is target

# 🔹 Encode target if it's not numeric
if df[target_col].dtype == "object":
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

X = df.drop(columns=[target_col])
y = df[target_col]

# 🔹 Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 Initialize classifiers
models = {
    "LR": LogisticRegression(max_iter=500),
    "RF": RandomForestClassifier(),
    "XGB": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}

print(f"\n🔍 Evaluating: {os.path.basename(filename)}")
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"✅ {name} Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"❌ {name} failed: {e}")

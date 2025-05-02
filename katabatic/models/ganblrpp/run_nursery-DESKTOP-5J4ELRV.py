import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from katabatic.models.ganblrpp.ganblrpp_adapter import GanblrppAdapter

print("\n[STEP 1] Loading dataset...")
df = pd.read_csv("katabatic/models/nursery.csv")
print(f"✅ Loaded dataset with shape: {df.shape}")

target_column = "Target"
X = df.drop(columns=[target_column])
y = df[target_column]

# Ordinal encode features and target
encoder_X = OrdinalEncoder()
X_encoded = encoder_X.fit_transform(X)

encoder_y = LabelEncoder()
y_encoded = encoder_y.fit_transform(y)

# Split real dataset (80/10/10 split)
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("✅ Training data shape:", X_train.shape)
print("✅ Validation (for synthetic generation):", X_valid.shape)
print("✅ Test set shape:", X_test.shape)

# Initialize GANBLR++ model
print("\n[STEP 2] Training GANBLR++ model...")
adapter = GanblrppAdapter(numerical_columns=[])  # no numerical features here
adapter.load_model()
adapter.fit(X_train, y_train, k=2, epochs=10, batch_size=64)
print("✅ GANBLR++ training complete.")

# Generate synthetic samples
print("\n[STEP 3] Generating synthetic samples...")
synthetic_df = adapter.generate(size=len(X_valid))
print("✅ Synthetic data shape:", synthetic_df.shape)

X_synth = synthetic_df.iloc[:, :-1]
y_synth = synthetic_df.iloc[:, -1].astype(int)

# Inject missing classes directly
real_classes = np.unique(y_valid)
synth_classes = np.unique(y_synth)
missing_classes = np.setdiff1d(real_classes, synth_classes)

if len(missing_classes) > 0:
    print(f"⚠️ Injecting missing class(es): {missing_classes}")
    for cls in missing_classes:
        # Randomly select one real instance from class `cls`
        indices = np.where(y_valid == cls)[0]
        if len(indices) > 0:
            selected_idx = np.random.choice(indices)
            X_synth = pd.concat([X_synth, pd.DataFrame([X_valid[selected_idx]])], ignore_index=True)
            y_synth = np.append(y_synth, cls)
# 🔧 Fix XGBoost object column error:
X_synth = X_synth.astype(int)
X_test = X_test.astype(int)

print("\n🔹 [STEP 4] TSTR Evaluation using synthetic data")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "MLP Classifier": MLPClassifier(max_iter=500),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

accuracies = {}
conf_matrices = {}

for name, clf in classifiers.items():
    print(f"\n🔸 Training {name}...")
    try:
        clf.fit(X_synth, y_synth)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        accuracies[name] = acc
        conf_matrices[name] = cm

        print(f"✅ {name} Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"❌ {name} failed:", str(e))

# Plot accuracy bar chart
plt.figure(figsize=(8,5))
plt.bar(accuracies.keys(), accuracies.values(), color='lightgreen')
plt.title("TSTR Classifier Accuracy (Synthetic Train → Real Test)")
plt.ylabel("Accuracy")
plt.ylim(0.0, 1.0)
plt.xticks(rotation=15)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("tstr_accuracy_bar_chart.png")
plt.show()

# Plot XGBoost confusion matrix if available
if "XGBoost" in conf_matrices:
    xgb_cm = conf_matrices["XGBoost"]
    plt.figure(figsize=(6,5))
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("XGBoost Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("xgboost_confusion_matrix.png")
    plt.show()
else:
    print("⚠️ Skipping XGBoost confusion matrix plot.")

print("\n✅ All done. TSTR plot and confusion matrix saved.")

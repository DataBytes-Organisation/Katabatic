import os
import numpy as np
import pandas as pd
from tabpfgen import TabPFGen
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

dataset_dir = "datasets"
results = []

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

for file in os.listdir(dataset_dir):
    if not file.endswith(".csv"):
        continue

    print(f"\n🔍 Processing: {file}")
    try:
        df = pd.read_csv(os.path.join(dataset_dir, file))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Normalize labels
        y = pd.Series(LabelEncoder().fit_transform(y))

        # Skip if too few classes
        if len(np.unique(y)) < 2:
            print(f"⚠️ Skipped {file} due to too few classes.")
            continue

        # Split real data
        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train on real data
        real_accuracies = {}
        for clf_name, clf in {
            "LR": LogisticRegression(max_iter=1000),
            "RF": RandomForestClassifier(),
            "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }.items():
            acc = evaluate_model(clf, X_train_real, y_train_real, X_test_real, y_test_real)
            real_accuracies[clf_name] = acc
            print(f"✅ {clf_name} | Real Acc: {acc:.4f}")

        # Generate synthetic data using TabPFGen with label injection
        generator = TabPFGen()
        X_synth_all, y_synth_all = [], []
        n_samples_per_class = 100

        for label in np.unique(y_train_real):
            X_class = X_train_real[y_train_real == label]
            y_class = y_train_real[y_train_real == label]

            if len(X_class) < 5:
                print(f"⚠️ Skipping label {label} (too few samples)")
                continue

            X_synth, y_synth = generator.generate_classification(
                X_class.values,
                y_class.values,
                n_samples=n_samples_per_class,
                class_label=label
            )
            X_synth_all.append(X_synth)
            y_synth_all.append(np.full(n_samples_per_class, label))

        if not X_synth_all:
            print(f"⚠️ Skipped {file} due to no valid synthetic data.")
            continue

        X_synth_total = np.vstack(X_synth_all)
        y_synth_total = np.concatenate(y_synth_all)

        # Train classifiers on synthetic data
        for clf_name, clf in {
            "LR": LogisticRegression(max_iter=1000),
            "RF": RandomForestClassifier(),
            "XGB": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }.items():
            acc = evaluate_model(clf, X_synth_total, y_synth_total, X_test_real, y_test_real)
            print(f"✅ {clf_name} | Synthetic Acc: {acc:.4f}")

            results.append({
                "Dataset": file,
                "Classifier": clf_name,
                "Accuracy_Real": real_accuracies[clf_name],
                "Accuracy_Synthetic": acc
            })

    except Exception as e:
        print(f"⚠️ Skipped {file} due to error: {e}")

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("benchmark_results_full.csv", index=False)

print("\n📋 Full Benchmark Results:\n")
print(df_results)
print("\n📊 All results saved to benchmark_results_full.csv")


# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("benchmark_results_final.csv", index=False)

# Print results
print("\n📋 Final Benchmark Results:\n")
print(df_results.to_string(index=False))
print("\n📊 All results saved to benchmark_results_final.csv")




# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("benchmark_results_full.csv", index=False)

# ✅ Print in console
print("\n📋 Full Benchmark Results:\n")
print(df_results.to_string(index=False))

print("\n📊 All results saved to benchmark_results_full.csv")

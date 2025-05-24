import openml
from sklearn.model_selection import train_test_split
import os
import pandas as pd

dataset_names = [
    "blood-transfusion-service-center",
    "kc1",
    "kr-vs-kp",
    "mfeat-factors",
    "phoneme",
    "qsar-biodeg",
    "wdbc",
    "wine"
]

os.makedirs("datasets", exist_ok=True)

for name in dataset_names:
    print(f"🔽 Downloading: {name}")
    dataset = openml.datasets.get_dataset(name)
    df, *_ = dataset.get_data()
    df.to_csv(f"datasets/{name}.csv", index=False)

print("✅ All datasets downloaded to 'datasets/' folder.")

# save as download_openml_datasets.py
from tabpfgen.benchmarking.datasets import load_classification_dataset

dataset_names = [
    "blood-transfusion-service-center",
    "kc1", "kr-vs-kp", "mfeat-factors",
    "phoneme", "qsar-biodeg", "wdbc", "wine"
]

for name in dataset_names:
    print(f"Downloading: {name}")
    X, y = load_classification_dataset(name)

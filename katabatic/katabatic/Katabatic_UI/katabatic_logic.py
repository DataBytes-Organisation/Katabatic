import pandas as pd
from katabatic.models.meg_DGEK.meg_adapter import MegAdapter
from katabatic.models.meg_DGEK.utils import get_demo_data
from sklearn.model_selection import train_test_split

# Load demo data
def load_demo_data():
    df = get_demo_data('adult-raw')
    return df

# Train the model
def train_model(data, test_size=0.5, epochs=5):
    x, y = data.values[:, :-1], data.values[:, -1]
    X_train, _, y_train, _ = train_test_split(x, y, test_size=test_size)
    adapter = MegAdapter()
    adapter.load_model()
    adapter.fit(X_train, y_train, epochs=epochs)
    return adapter

# Generate synthetic data
def generate_data(adapter, size=5):
    synthetic_data = adapter.generate(size=size)
    return synthetic_data
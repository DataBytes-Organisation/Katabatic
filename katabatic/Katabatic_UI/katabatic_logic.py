import pandas as pd
from sklearn.metrics import accuracy_score
from katabatic.models.meg_DGEK.meg_adapter import MegAdapter
from katabatic.models.meg_DGEK.utils import get_demo_data
from sklearn.model_selection import train_test_split

# Load data from a user-uploaded file or use demo data
def load_data(data_pathname=None):
    """
    Load data from the specified pathname or use demo data if no pathname is provided.

    Args:
        data_pathname (str): Path to the user-uploaded dataset.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    if data_pathname:
        print(f"[INFO] Loading user dataset from {data_pathname}")
        try:
            data = pd.read_csv(data_pathname)
            print("[SUCCESS] User dataset loaded successfully.")
            return data
        except FileNotFoundError:
            print(f"[ERROR] File '{data_pathname}' not found.")
        except pd.errors.EmptyDataError:
            print("[ERROR] The file is empty.")
        except pd.errors.ParserError:
            print("[ERROR] Parsing error while reading the file.")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error
    else:
        print("[INFO] Loading demo dataset 'adult-raw'")
        return get_demo_data('adult-raw')

# Train the model
def train_model(data, test_size=0.5, epochs=5):
    """
    Train the MEG model using the given data.

    Args:
        data (pd.DataFrame): Input dataset.
        test_size (float): Fraction of the dataset to use as the test set.
        epochs (int): Number of epochs for training the model.

    Returns:
        tuple: (trained adapter, reliability score as a percentage).
    """
    x, y = data.values[:, :-1], data.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    # Initialize and load the adapter
    adapter = MegAdapter()
    adapter.load_model()
    adapter.fit(X_train, y_train, epochs=epochs)

    # Calculate reliability
    try:
        if hasattr(adapter, 'predict'):
            y_pred = adapter.predict(X_test)
            reliability = round(accuracy_score(y_test, y_pred) * 100, 2)
        else:
            reliability = 0.0  # Default reliability if predictions are unavailable
    except Exception as e:
        print(f"[ERROR] Failed to calculate reliability: {e}")
        reliability = 0.0

    return adapter, reliability

# Generate synthetic data
def generate_data(adapter, size=5):
    """
    Generate synthetic data using the trained adapter.

    Args:
        adapter (MegAdapter): Trained MEG adapter.
        size (int): Number of synthetic samples to generate.

    Returns:
        pd.DataFrame: Generated synthetic dataset.
    """
    try:
        synthetic_data = adapter.generate(size=size)
        return synthetic_data
    except Exception as e:
        print(f"[ERROR] Failed to generate synthetic data: {e}")
        return pd.DataFrame()
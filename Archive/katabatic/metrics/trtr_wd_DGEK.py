from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import wasserstein_distance
import numpy as np

def compute_wd(X_real, X_synthetic):
    """
    Compute Wasserstein Distance between real and synthetic datasets.

    Parameters:
    - X_real: np.ndarray, real dataset features
    - X_synthetic: np.ndarray, synthetic dataset features

    Returns:
    - wd: float, Wasserstein Distance value
    """
    # Ensure data is in 1D arrays
    X_real = np.ravel(X_real)
    X_synthetic = np.ravel(X_synthetic)
    
    # Compute Wasserstein Distance
    wd = wasserstein_distance(X_real, X_synthetic)
    return wd

def evaluate(X_synthetic, y_synthetic, X_real, y_real):
    """
    Evaluate the Wasserstein Distance metric between synthetic and real datasets.

    Parameters:
    - X_synthetic: np.ndarray, synthetic dataset features
    - y_synthetic: np.ndarray, synthetic dataset labels
    - X_real: np.ndarray, real dataset features
    - y_real: np.ndarray, real dataset labels

    Returns:
    - wd: float, Wasserstein Distance value between the real and synthetic datasets
    """
    # Convert to numpy arrays
    X_synthetic = X_synthetic.to_numpy()
    y_synthetic = y_synthetic.to_numpy().ravel()
    X_real = X_real.to_numpy()
    y_real = y_real.to_numpy().ravel()
    
    # Combine features and labels
    X_real_combined = np.hstack([X_real, y_real.reshape(-1, 1)])
    X_synthetic_combined = np.hstack([X_synthetic, y_synthetic.reshape(-1, 1)])
    
    # Compute Wasserstein Distance
    wd_value = compute_wd(X_real_combined, X_synthetic_combined)
    
    return wd_value

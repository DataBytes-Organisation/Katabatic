from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import jensenshannon
import numpy as np

def compute_jsd(X_real, X_synthetic):
    """
    Compute Jensen-Shannon Divergence between real and synthetic datasets.

    Parameters:
    - X_real: np.ndarray, real dataset features
    - X_synthetic: np.ndarray, synthetic dataset features

    Returns:
    - jsd: float, Jensen-Shannon Divergence value
    """
    # Ensure data is in 1D arrays
    X_real = np.ravel(X_real)
    X_synthetic = np.ravel(X_synthetic)
    
    # Compute histograms
    hist_real, _ = np.histogram(X_real, bins=np.arange(X_real.max() + 2) - 0.5, density=True)
    hist_synthetic, _ = np.histogram(X_synthetic, bins=np.arange(X_synthetic.max() + 2) - 0.5, density=True)
    
    # Handle empty histograms
    if np.all(hist_real == 0):
        hist_real += 1e-10
    if np.all(hist_synthetic == 0):
        hist_synthetic += 1e-10
    
    # Compute Jensen-Shannon Divergence
    jsd = jensenshannon(hist_real, hist_synthetic)
    return jsd

def evaluate(X_synthetic, y_synthetic, X_real, y_real):
    """
    Evaluate the Jensen-Shannon Divergence metric between synthetic and real datasets.

    Parameters:
    - X_synthetic: np.ndarray, synthetic dataset features
    - y_synthetic: np.ndarray, synthetic dataset labels
    - X_real: np.ndarray, real dataset features
    - y_real: np.ndarray, real dataset labels

    Returns:
    - jsd: float, Jensen-Shannon Divergence value between the real and synthetic datasets
    """
    # Convert to numpy arrays
    X_synthetic = X_synthetic.to_numpy()
    y_synthetic = y_synthetic.to_numpy().ravel()
    X_real = X_real.to_numpy()
    y_real = y_real.to_numpy().ravel()
    
    # Combine features and labels
    X_real_combined = np.hstack([X_real, y_real.reshape(-1, 1)])
    X_synthetic_combined = np.hstack([X_synthetic, y_synthetic.reshape(-1, 1)])
    
    # Compute JSD
    jsd_value = compute_jsd(X_real_combined, X_synthetic_combined)
    
    return jsd_value


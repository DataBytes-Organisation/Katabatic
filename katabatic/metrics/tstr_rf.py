from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate(X_synthetic, y_synthetic, X_real, y_real):
    """
    Evaluate the quality of synthetic data using TSTR (Train on Synthetic, Test on Real) method.

    This function trains a RandomForestClassifier model on synthetic data and evaluates its
    performance on real data. The accuracy score is returned as a metric for model performance.

    Parameters:
    - X_synthetic: Features of the synthetic data
    - y_synthetic: Labels of the synthetic data
    - X_real: Features of the real data
    - y_real: Labels of the real data

    Returns:
    - accuracy: Accuracy score of the RandomForestClassifier model on the real data.
    """

    # Data Validation
    if X_synthetic.shape[1] != X_real.shape[1]:
        raise ValueError(
            "The number of features in X_synthetic and X_real must be the same."
        )

    if len(y_synthetic) != len(X_synthetic):
        raise ValueError(
            "X_synthetic and y_synthetic must have the same number of samples."
        )

    if len(y_real) != len(X_real):
        raise ValueError("X_real and y_real must have the same number of samples.")

    # Convert DataFrames to NumPy arrays if necessary
    X_synthetic = (
        X_synthetic.to_numpy() if hasattr(X_synthetic, "to_numpy") else X_synthetic
    )
    y_synthetic = (
        y_synthetic.to_numpy().ravel()
        if hasattr(y_synthetic, "to_numpy")
        else y_synthetic
    )
    X_real = X_real.to_numpy() if hasattr(X_real, "to_numpy") else X_real
    y_real = y_real.to_numpy().ravel() if hasattr(y_real, "to_numpy") else y_real

    # Combine y_synthetic and y_real to ensure consistent label encoding
    y_combined = np.concatenate([y_synthetic, y_real])

    # Encode labels
    le = LabelEncoder()
    le.fit(y_combined)  # Ensure all labels from both datasets are accounted for
    y_synthetic = le.transform(y_synthetic)
    y_real = le.transform(y_real)

    # TSTR Evaluation using RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_synthetic, y_synthetic)
    y_pred = model.predict(X_real)

    # Return the accuracy score
    return accuracy_score(y_real, y_pred)

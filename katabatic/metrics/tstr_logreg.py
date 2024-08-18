from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np


def evaluate(X_synthetic, y_synthetic, X_real, y_real):
    """
    Evaluate the quality of synthetic data using TSTR (Train on Synthetic, Test on Real) method.

    This function trains a Logistic Regression model on synthetic data and evaluates its performance
    on a test set derived from real data. The accuracy score is returned as a metric
    for model performance.

    Parameters:
    - X_synthetic: Features of the synthetic data
    - y_synthetic: Labels of the synthetic data
    - X_real: Features of the real data
    - y_real: Labels of the real data

    Returns:
    - accuracy: Accuracy score of the Logistic Regression model on the real test set.
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

    # Split the real data into training and testing sets
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
        X_real, y_real, test_size=0.33, random_state=42
    )

    # Encode labels (necessary if labels are not numerical)
    le = LabelEncoder()
    y_synthetic = le.fit_transform(y_synthetic)
    y_test_real = le.transform(y_test_real)

    # TSTR Evaluation using Logistic Regression
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_synthetic, y_synthetic)
    y_pred = model.predict(X_test_real)

    # Return the accuracy score
    return accuracy_score(y_test_real, y_pred)

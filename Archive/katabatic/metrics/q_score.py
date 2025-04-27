from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


def evaluate(X_synthetic, y_synthetic, X_real, y_real, use_labels=False):
    """
    Evaluate the quality of synthetic data using a Q-score approach.

    This method evaluates the quality of synthetic data by assessing how distinguishable
    the synthetic data is from the real data. If `use_labels` is True, the original labels
    (`y_real` and `y_synthetic`) are included as additional features to improve the accuracy
    of this distinction. If `use_labels` is False, only the features are used.

    Parameters:
    - X_synthetic: Features of the synthetic data
    - y_synthetic: Labels of the synthetic data
    - X_real: Features of the real data
    - y_real: Labels of the real data
    - use_labels: Boolean flag to determine whether to use labels in evaluation.

    Returns:
    - q_score: AUC score indicating how well the model can distinguish between real and synthetic data.
    """

    # Convert to numpy arrays if they are in DataFrame format
    X_synthetic = (
        X_synthetic.to_numpy() if isinstance(X_synthetic, pd.DataFrame) else X_synthetic
    )
    y_synthetic = (
        y_synthetic.to_numpy().ravel()
        if isinstance(y_synthetic, pd.DataFrame)
        else y_synthetic
    )
    X_real = X_real.to_numpy() if isinstance(X_real, pd.DataFrame) else X_real
    y_real = y_real.to_numpy().ravel() if isinstance(y_real, pd.DataFrame) else y_real

    # Identify categorical features and numeric features
    categorical_features = np.where(X_synthetic.dtype == "O")[
        0
    ]  # Assuming all object columns are categorical
    numeric_features = np.where(X_synthetic.dtype != "O")[0]

    # Define the preprocessing pipeline for the features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    if use_labels:
        # Combine the features and labels of the real and synthetic data
        X_combined = np.vstack([X_real, X_synthetic])
        y_combined = np.hstack([y_real, y_synthetic])

        # Create labels for distinguishing between real and synthetic data: 1 for real, 0 for synthetic
        real_vs_synthetic_labels = np.hstack(
            [np.ones(X_real.shape[0]), np.zeros(X_synthetic.shape[0])]
        )

        # Combine the real_vs_synthetic labels with the original labels as additional features
        X_combined_with_labels = np.column_stack((X_combined, y_combined))

        # Create a pipeline that applies the preprocessor and then splits the data
        X_combined_with_labels = preprocessor.fit_transform(X_combined_with_labels)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined_with_labels,
            real_vs_synthetic_labels,
            test_size=0.33,
            random_state=42,
        )
    else:
        # Use only features for the evaluation
        X_combined = np.vstack([X_real, X_synthetic])

        # Create labels for distinguishing between real and synthetic data: 1 for real, 0 for synthetic
        real_vs_synthetic_labels = np.hstack(
            [np.ones(X_real.shape[0]), np.zeros(X_synthetic.shape[0])]
        )

        # Preprocess the combined data
        X_combined = preprocessor.fit_transform(X_combined)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, real_vs_synthetic_labels, test_size=0.33, random_state=42
        )

    # Train a RandomForest model to distinguish between real and synthetic data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of being real

    # Calculate AUC as Q-score
    q_score = roc_auc_score(y_test, y_pred_proba)

    return q_score


"""
**Pros:**
    - **Comprehensive Evaluation (use_labels=True):** By including both features and labels,
      the method provides a more complete assessment of how well the synthetic data mimics
      the real data.
    - **Simple Approach (use_labels=False):** For cases where the labels might not be important,
      this simpler method can still provide valuable insights.
    - **Quantitative Metric (Q-score):** The use of AUC as the Q-score provides a quantitative
      measure of similarity, where a lower score (closer to 0.5) indicates higher similarity.

    **Cons:**
    - **Label Dependency (use_labels=True):** The effectiveness of the comprehensive method
      is dependent on the quality and relevance of the labels. If labels are noisy or poorly defined,
      this can negatively impact the Q-score.
    - **Model Complexity (use_labels=True):** Adding labels as features increases model complexity,
      requiring more computational resources and potentially leading to overfitting.

    **When to Use This Approach:**
    - **use_labels=True:** Use this approach when generating synthetic data for classification
      or regression tasks where the labels are significant, and preserving the relationship
      between features and labels is crucial.
    - **use_labels=False:** Use this simpler approach when you are primarily concerned with
      feature distributions and not the relationship between features and labels.
"""

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import pairwise_distances, jaccard_score
from scipy.stats import ks_2samp, wasserstein_distance
import numpy as np
import pandas as pd


def evaluate(X_synthetic, X_real, similarity_metric="ks_test"):
    """
    Evaluate the similarity between real and synthetic data using a specified metric.

    This function compares the real and synthetic data using the specified similarity metric.

    Parameters:
    - X_synthetic: Features of the synthetic data.
    - X_real: Features of the real data.
    - similarity_metric: The similarity metric to apply. Supported options are:
        - 'ks_test': Kolmogorov-Smirnov test for continuous distributions.
        - 'wasserstein': Wasserstein distance for continuous distributions.
        - 'jaccard': Jaccard similarity for binary or categorical data.
        - 'euclidean': Euclidean distance between datasets.
        - 'cosine': Cosine similarity between datasets.

    Returns:
    - similarity_score: The computed similarity score based on the specified metric.
    """

    # Data Validation
    if X_synthetic.shape[1] != X_real.shape[1]:
        raise ValueError(
            "The number of features in X_synthetic and X_real must be the same."
        )

    # Convert DataFrames to NumPy arrays if necessary
    X_synthetic = (
        X_synthetic.to_numpy() if hasattr(X_synthetic, "to_numpy") else X_synthetic
    )
    X_real = X_real.to_numpy() if hasattr(X_real, "to_numpy") else X_real

    # Define the preprocessing pipeline for the features
    categorical_features = np.where(X_synthetic.dtype == "O")[
        0
    ]  # Assuming all object columns are categorical
    numeric_features = np.where(X_synthetic.dtype != "O")[0]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Apply preprocessing to both real and synthetic data
    X_synthetic_processed = preprocessor.fit_transform(X_synthetic)
    X_real_processed = preprocessor.transform(X_real)

    # Compute similarity based on the specified metric
    if similarity_metric == "ks_test":
        similarity_scores = [
            ks_2samp(X_real_processed[:, i], X_synthetic_processed[:, i])[0]
            for i in range(X_real_processed.shape[1])
        ]
        similarity_score = np.mean(similarity_scores)

    elif similarity_metric == "wasserstein":
        similarity_scores = [
            wasserstein_distance(X_real_processed[:, i], X_synthetic_processed[:, i])
            for i in range(X_real_processed.shape[1])
        ]
        similarity_score = np.mean(similarity_scores)

    elif similarity_metric == "jaccard":
        jaccard_scores = []
        for i in range(X_real_processed.shape[1]):
            if (
                len(np.unique(X_real_processed[:, i])) <= 20
            ):  # Assume categorical data if low cardinality
                jaccard_scores.append(
                    jaccard_score(
                        X_real_processed[:, i],
                        X_synthetic_processed[:, i],
                        average="macro",
                    )
                )
        similarity_score = np.mean(jaccard_scores)

    elif similarity_metric == "euclidean":
        similarity_score = np.mean(
            pairwise_distances(
                X_real_processed, X_synthetic_processed, metric="euclidean"
            )
        )

    elif similarity_metric == "cosine":
        similarity_score = np.mean(
            1
            - pairwise_distances(
                X_real_processed, X_synthetic_processed, metric="cosine"
            )
        )

    else:
        raise ValueError(f"Unsupported similarity metric: {similarity_metric}")

    return similarity_score

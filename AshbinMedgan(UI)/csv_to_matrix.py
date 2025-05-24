import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

REQUIRED_DIM = 294

def csv_to_matrix(csv_path, save_feature_names_path="column_names.csv"):
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_path)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Convert all columns to string for one-hot encoding
    df = df.astype(str)

    # One-hot encoding
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    matrix = encoder.fit_transform(df)

    # Get all feature names
    all_feature_names = encoder.get_feature_names_out(df.columns)

    # Adjust matrix to exactly REQUIRED_DIM
    current_dim = matrix.shape[1]
    if current_dim < REQUIRED_DIM:
        # Pad with zeros
        padding = np.zeros((matrix.shape[0], REQUIRED_DIM - current_dim))
        matrix = np.hstack((matrix, padding))
        feature_names = list(all_feature_names) + [f"pad_{i}" for i in range(REQUIRED_DIM - current_dim)]
    elif current_dim > REQUIRED_DIM:
        # Trim extra columns
        matrix = matrix[:, :REQUIRED_DIM]
        feature_names = list(all_feature_names[:REQUIRED_DIM])
    else:
        feature_names = list(all_feature_names)

    # Save column names for reverse mapping
    pd.DataFrame(feature_names, columns=["Feature"]).to_csv(save_feature_names_path, index=False)

    return matrix

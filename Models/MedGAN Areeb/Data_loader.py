import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Loader for car.csv (all categorical)
def load_car_data(csv_path, test_size=0.2, n_shuffle=5):
    """
    Load and preprocess the Car dataset (categorical features + label in last column).

    Args:
        csv_path (str): Path to the dataset CSV file.
        test_size (float): Fraction of the dataset to use for testing.
        n_shuffle (int): Number of times to randomly shuffle the data before splitting.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(csv_path, header=None)

    # Separate features and labels
    X_real = df.iloc[:, :-1]
    y_real = df.iloc[:, -1]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_real)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include='number').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', MinMaxScaler(), numerical_cols)

        ]
    )

    X_processed = preprocessor.fit_transform(X_real)

    # Shuffle if needed
    for _ in range(n_shuffle):
        shuffled_indices = np.random.permutation(len(X_processed))
        X_processed = X_processed[shuffled_indices]
        y_encoded = y_encoded[shuffled_indices]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Loader for Bank_Personal_Loan.csv (categorical + numeric)
def load_bank_data(csv_path, test_size=0.2, n_shuffle=5):
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    X_real = df.iloc[:, :-1]
    y_real = df.iloc[:, -1]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_real)
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include='number').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', MinMaxScaler(), numerical_cols)

        ]
    )

    X_processed = preprocessor.fit_transform(X_real)

    # Shuffle if needed
    for _ in range(n_shuffle):
        shuffled_indices = np.random.permutation(len(X_processed))
        X_processed = X_processed[shuffled_indices]
        y_encoded = y_encoded[shuffled_indices]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


# Loader for satellite.csv (all numeric)
def load_satellite_data(csv_path, test_size=0.2, n_shuffle=5):
    df = pd.read_csv(csv_path)

    # Assume last column is the label
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    scaler = MinMaxScaler()
    data = scaler.fit_transform(X)

    for _ in range(n_shuffle):
        shuffled_indices = np.random.permutation(len(data))
        data = data[shuffled_indices]
        y = y.iloc[shuffled_indices]

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

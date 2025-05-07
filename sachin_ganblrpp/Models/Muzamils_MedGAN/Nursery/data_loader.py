import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_shuttle_data(csv_path, test_size=0.2, normalize=True, n_shuffle=5):
    """
    Load and preprocess the Shuttle dataset with multiple randomizations.

    Args:
        csv_path (str): Path to the dataset CSV file.
        test_size (float): Fraction of the dataset to use for testing.
        normalize (bool): Whether to scale features between 0 and 1.
        n_shuffle (int): Number of times to randomly shuffle the data before splitting.

    Returns:
        tuple: (X_train, X_test) as NumPy arrays
    """
    df = pd.read_csv(csv_path, header=None)  # No header in shuttle.csv
    df = df.drop(columns=[df.columns[-1]])   # Drop label/class column

    if normalize:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
    else:
        data = df.values

    # Shuffle the data n_shuffle times
    for _ in range(n_shuffle):
        np.random.shuffle(data)

    X_train, X_test = train_test_split(data, test_size=test_size, random_state=None)
    return X_train, X_test

    
    def load_nursery_data(csv_path, test_size=0.2):
    df = pd.read_csv(csv_path)

    # Separate label and features
    if 'Target' not in df.columns:
        raise ValueError("Expected label column named 'Target'")
    
    y = df['Target']
    df = df.drop(columns=['Target'])

    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols = df.select_dtypes(include='number').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols),
            ('num', MinMaxScaler(), numerical_cols)
        ]
    )

    processed_data = preprocessor.fit_transform(df)
    X_train, X_test, y_train, y_test = train_test_split(processed_data, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def load_letter_data(csv_path, test_size=0.2, normalize=True):
    """
    Load and preprocess the Letter Recognition dataset for MedGAN.

    Args:
        csv_path (str): Path to the dataset CSV file.
        test_size (float): Fraction of the dataset to use for testing.
        normalize (bool): Whether to scale features between 0 and 1.

    Returns:
        tuple: (X_train, X_test) as NumPy arrays
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    df = pd.read_csv(csv_path)

    # Drop the label column
    if 'letter' in df.columns:
        df = df.drop(columns=['letter'])

    # Normalize features
    if normalize:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df)
    else:
        data = df.values

    X_train, X_test = train_test_split(data, test_size=test_size, random_state=42)
    return X_train, X_test

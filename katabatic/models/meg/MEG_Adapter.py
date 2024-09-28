from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .meg import MEG

class MegAdapter(KatabaticModelSPI):
    """
    Adapter class for the MEG model to interface with KatabaticModelSPI.

    Attributes:
        type (str): The type of model, either 'discrete' or 'continuous'.
        candidate_labels (list): The candidate labels for the model.
        batch_size (int): The batch size for training the model.
        epochs (int): The number of epochs for training the model.
        training_sample_size (int): The size of the training sample.
    """

    def __init__(self, type="discrete", candidate_labels=None):
        """
        Initialize the MEG Adapter with the specified type and candidate labels.

        Args:
            type (str): The type of model, either 'discrete' or 'continuous'. Default is 'discrete'.
            candidate_labels (list): The candidate labels for the model. Default is None.
        """
        self._d = None
        self.batch_size = None
        self.epochs = None
        self.k = None
        self._units = None
        self._candidate_labels = candidate_labels
        self._column_names = None
        self._weight = None
        self._ordinal_encoder = None
        self.training_sample_size = 0
        self.model = None

    def load_model(self):
        """
        Initialize and load the MEG model.

        Returns:
            MEG: An instance of the MEG model.
        """
        print("[INFO] Initializing MEG Model")
        self.model = MEG()
        return self.model
    
    def preprocess_data(self, X, y):
        """
        Preprocess data by converting categorical features and labels to numerical format.

        Args:
            X (pd.DataFrame or np.ndarray): Features to preprocess.
            y (pd.Series or np.ndarray): Labels to preprocess.

        Returns:
            Tuple: (preprocessed_X, preprocessed_y)
        """
        # Convert X to DataFrame if it's not already
        if isinstance(X, np.ndarray):
            if self._column_names is None:
                self._column_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self._column_names)

        print("[INFO] Initial X columns:", X.columns)

        # Handle categorical features
        for col in X.select_dtypes(include=['object']).columns:
            print(f"[INFO] Converting column '{col}' to numerical format")
            X[col] = X[col].astype('category').cat.codes

        # Handle categorical labels
        if y is not None:
            if isinstance(y, np.ndarray):
                y = pd.Series(y)
            if y.dtype == 'object':
                print(f"[INFO] Converting labels to numerical format")
                y = y.astype('category').cat.codes

        return X, y

    def load_data(self, data_pathname):
        """
        Load data from the specified pathname.

        Args:
            data_pathname (str): The path to the data file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        print(f"[INFO] Loading data from {data_pathname}")
        try:
            data = pd.read_csv(data_pathname)
            print("[SUCCESS] Data loaded successfully.")
            return data
        except FileNotFoundError:
            print(f"[ERROR] File '{data_pathname}' not found.")
        except pd.errors.EmptyDataError:
            print("[ERROR] The file is empty.")
        except pd.errors.ParserError:
            print("[ERROR] Parsing error while reading the file.")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
            return None

    def fit(self, X_train, y_train=None, k=0, batch_size=32, epochs=10):
        """
        Train the MEG model on the input data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels. Default is None.
            k (int): An optional parameter for the model's fit method. Default is 0.
            batch_size (int): The batch size for training. Default is 32.
            epochs (int): The number of epochs for training. Default is 10.
        """
        if self.model is None:
            print("[ERROR] Model is not loaded. Call 'load_model()' first.")
            return

        # Ensure X_train and y_train are pandas DataFrames or Series
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=self._column_names)
        if isinstance(y_train, np.ndarray):
            y_train = pd.Series(y_train)

        # Preprocess data
        X_train, y_train = self.preprocess_data(X_train, y_train)

        try:
            print("[INFO] Training MEG model")
            self.model.fit(
                X_train, y_train, k, batch_size=batch_size, epochs=epochs, verbose=0
            )
            self.training_sample_size = len(X_train)
            print("[SUCCESS] Model training completed")
        except ValueError as e:
            print(f"[ERROR] ValueError during model training: {e}")
        except TypeError as e:
            print(f"[ERROR] TypeError during model training: {e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during model training: {e}")

    def generate(self, size=None):
        """
        Generate data using the MEG model.

        Args:
            size (int): The number of samples to generate. Defaults to the training sample size if not specified.

        Returns:
            pd.DataFrame or np.ndarray: The generated data.
        """
        if self.model is None:
            print("[ERROR] Model is not loaded. Call 'load_model()' first.")
            return None

        try:
            print("[INFO] Generating data using MEG model")
            if size is None:
                size = self.training_sample_size

            generated_data = self.model.sample(size, verbose=0)
            if isinstance(generated_data, np.ndarray):
                generated_data = pd.DataFrame(generated_data, columns=self._column_names)
            print("[SUCCESS] Data generation completed")
            return generated_data
        except ValueError as e:
            print(f"[ERROR] ValueError during data generation: {e}")
        except TypeError as e:
            print(f"[ERROR] TypeError during data generation: {e}")
        except AttributeError as e:
            print(f"[ERROR] AttributeError during data generation: {e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during data generation: {e}")
            return None




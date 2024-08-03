from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
from .ganblr import GANBLR


class GanblrAdapter(KatabaticModelSPI):
    """
    Adapter class for GANBLR model to interface with KatabaticModelSPI.

    Attributes:
        type (str): The type of model, either 'discrete' or 'continuous'.
        constraints: The constraints for the model (currently not used).
        batch_size (int): The batch size for training the model.
        epochs (int): The number of epochs for training the model.
        training_sample_size (int): The size of the training sample.
    """

    def __init__(self, type="discrete"):
        """
        Initialize the GANBLR Adapter with the specified type.

        Args:
            type (str): The type of model, either 'discrete' or 'continuous'. Default is 'discrete'.
        """
        self.type = type
        self.constraints = None
        self.batch_size = None
        self.epochs = None

    def load_model(self):
        """
        Initialize and load the GANBLR model.

        Returns:
            GANBLR: An instance of the GANBLR model.
        """
        print("[INFO] Initializing GANBLR Model")
        self.model = GANBLR()
        self.training_sample_size = 0
        return self.model

    def load_data(self, data_pathname):
        """
        Load data from the specified pathname.

        Args:
            data_pathname (str): The path to the data file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            FileNotFoundError: If the file specified by data_pathname is not found.
            pd.errors.EmptyDataError: If the file is empty.
            pd.errors.ParserError: If there is a parsing error while reading the file.
            Exception: For any other exceptions that occur during data loading.
        """
        print(f"[INFO] Loading data from {data_pathname}")
        try:
            data = pd.read_csv(data_pathname)
            print("[SUCCESS] Data loaded successfully.")
        except FileNotFoundError:
            print(f"[ERROR] File '{data_pathname}' not found.")
        except pd.errors.EmptyDataError:
            print("[ERROR] The file is empty.")
        except pd.errors.ParserError:
            print("[ERROR] Parsing error while reading the file.")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred: {e}")
        return data

    def fit(self, X_train, y_train, k=0, epochs=10, batch_size=64):
        """
        Train the GANBLR model on the input data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels.
            k (int): An optional parameter for the model's fit method. Default is 0.
            epochs (int): The number of epochs for training. Default is 10.
            batch_size (int): The batch size for training. Default is 64.

        Raises:
            ValueError: If there is a value error such as wrong input shape.
            TypeError: If there is a type error such as wrong data type.
            Exception: For any other exceptions that occur during training.
        """
        try:
            print("[INFO] Training GANBLR model")
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
        Generate data using the GANBLR model.

        Args:
            size (int): The number of samples to generate. If not specified, defaults to the training sample size.

        Returns:
            pd.DataFrame or np.ndarray: The generated data.

        Raises:
            ValueError: If there is a value error such as invalid size.
            TypeError: If there is a type error such as wrong data type for size.
            AttributeError: If the model does not have a sample method.
            Exception: For any other exceptions that occur during data generation.
        """
        try:
            print("[INFO] Generating data using GANBLR model")
            if size is None:
                size = self.training_sample_size

            generated_data = self.model.sample(size, verbose=0)
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

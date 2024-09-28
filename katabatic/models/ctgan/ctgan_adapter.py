
from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import torch as torch
from .ctgan import CTGAN

class CtganAdapter(KatabaticModelSPI):
    def __init__(self,
                 embedding_dim=128,
                 generator_dim=(256, 256),
                 discriminator_dim=(256, 256),
                 generator_lr=2e-4,
                 generator_decay=1e-6,
                 discriminator_lr=2e-4,
                 discriminator_decay=1e-6,
                 batch_size=500,
                 discriminator_steps=1,
                 log_frequency=True,
                 verbose=False,
                 epochs=300,
                 pac=10,
                 cuda=True,
                 discrete_columns=None):
        assert batch_size % 2 == 0
        self.discrete_columns = discrete_columns
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None

    def load_model(self):
        """
        Initialize and load the CTGAN model.

        Returns:
            CTGAN: An instance of the CTGAN model.
        """
        print("[INFO] Initializing CTGAN Model")
        self.model = CTGAN(
            embedding_dim=self._embedding_dim,
            generator_dim=self._generator_dim,
            discriminator_dim=self._discriminator_dim,
            generator_lr=self._generator_lr,
            generator_decay=self._generator_decay,
            discriminator_lr=self._discriminator_lr,
            discriminator_decay=self._discriminator_decay,
            batch_size=self._batch_size,
            discriminator_steps=self._discriminator_steps,
            log_frequency=self._log_frequency,
            verbose=self._verbose,
            epochs=self._epochs,
            pac=self.pac,
            cuda=self._device.type == 'cuda'
        )
        self.training_sample_size = 0
        return self.model

    def load_data(self, data_pathname):
        """
        Load data from the specified pathname.

        Args:
            data_pathname (str): The path to the data file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame, or None if loading fails.
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
    
    def fit(self, X_train, y_train=None, discrete_columns=None, epochs=10):
        """
        Train the CTGAN model on the input data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray): Training labels (optional, not used here).
            discrete_columns (list): List of discrete columns.
            epochs (int): The number of epochs for training. Default is 10.

        Raises:
            ValueError: If there is a value error such as wrong input shape.
            TypeError: If there is a type error such as wrong data type.
            Exception: For any other exceptions that occur during training.
        """
        self.discrete_columns = discrete_columns if discrete_columns is not None else self.discrete_columns
        try:
            print("[INFO] Training CTGAN model")
            self.model.fit(X_train, discrete_columns=self.discrete_columns, epochs=epochs)
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

            generated_data = self.model.sample(size, condition_column=None, condition_value=None)
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


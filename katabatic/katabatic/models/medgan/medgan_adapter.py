from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from .medgan import MedGAN


class MedganAdapter(KatabaticModelSPI):
    """
    Adapter class for MedGAN model to interface with KatabaticModelSPI.

    Attributes:
        type (str): The type of model, either 'discrete' or 'continuous'.
        batch_size (int): The batch size for training the model.
        epochs (int): The number of epochs for training the model.
        training_sample_size (int): The size of the training sample.
    """

    def __init__(self, type="discrete"):
        """
        Initialize the MedGAN Adapter with the specified type.

        Args:
            type (str): The type of model, either 'discrete' or 'continuous'. Default is 'discrete'.
        """
        self.type = type
        self.batch_size = None
        self.epochs = None
        self.training_sample_size = None
        self.model = None
        self.feature_encoder = None
        self.label_encoder = None

    def load_model(self):
        """
        Initialize and load the MedGAN model.

        Returns:
            MedGAN: An instance of the MedGAN model.
        """
        print("[INFO] Initializing MedGAN Model")
        self.model = MedGAN()
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

    def preprocess_data(self, X, y):
        """
        Preprocess the data to encode categorical variables.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series or np.ndarray): The labels.

        Returns:
            np.ndarray, np.ndarray: The preprocessed features and labels.
        """
        print("[INFO] Preprocessing data...")

        # Encode features
        self.feature_encoder = OrdinalEncoder()
        X_processed = self.feature_encoder.fit_transform(X)

        # Encode labels
        if y is not None:
            self.label_encoder = LabelEncoder()
            y_processed = self.label_encoder.fit_transform(y)
        else:
            y_processed = None

        print("[SUCCESS] Data preprocessing completed.")
        return X_processed, y_processed

    def postprocess_data(self, X, y=None):
        """
        Post-process the generated data to decode categorical variables.

        Args:
            X (np.ndarray): The generated features.
            y (np.ndarray, optional): The generated labels.

        Returns:
            pd.DataFrame: The post-processed features and labels.
        """
        print("[INFO] Post-processing generated data...")

        # Decode features
        X_decoded = self.feature_encoder.inverse_transform(X)

        # Decode labels
        if y is not None:
            y_decoded = self.label_encoder.inverse_transform(y)
            print("[SUCCESS] Data post-processing completed.")
            return pd.DataFrame(
                X_decoded, columns=self.feature_encoder.feature_names_in_
            ), pd.Series(y_decoded)
        else:
            print("[SUCCESS] Data post-processing completed.")
            return pd.DataFrame(
                X_decoded, columns=self.feature_encoder.feature_names_in_
            )

    def fit(
        self,
        X_train,
        y_train=None,
        embedding_dim=128,
        random_dim=128,
        generator_dims=(128, 128),
        discriminator_dims=(256, 128, 1),
        compress_dims=(),
        decompress_dims=(),
        bn_decay=0.99,
        l2_scale=0.001,
        data_type="binary",
        epochs=10,
        batch_size=64,
    ):
        """
        Train the MedGAN model on the input data.

        Args:
            X_train (pd.DataFrame or np.ndarray): Training features.
            y_train (pd.Series or np.ndarray, optional): Training labels (not used in MedGAN).
            embedding_dim (int): Dimension of the embedding vector.
            random_dim (int): Dimension of the random noise vector.
            generator_dims (tuple): Dimensions of the generator layers.
            discriminator_dims (tuple): Dimensions of the discriminator layers.
            compress_dims (tuple): Dimensions of the compression layers in the autoencoder.
            decompress_dims (tuple): Dimensions of the decompression layers in the autoencoder.
            bn_decay (float): Decay rate for batch normalization.
            l2_scale (float): L2 regularization scale.
            data_type (str): Type of data ('binary' or 'count').
            epochs (int): The number of epochs for training. Default is 10.
            batch_size (int): The batch size for training. Default is 64.

        Raises:
            ValueError: If there is a value error such as wrong input shape.
            TypeError: If there is a type error such as wrong data type.
            Exception: For any other exceptions that occur during training.
        """
        try:
            print("[INFO] Preprocessing and training MedGAN model")
            X_train_processed, y_train_processed = self.preprocess_data(
                X_train, y_train
            )
            input_dim = X_train_processed.shape[1]
            self.model.fit(
                data=X_train_processed,
                input_dim=input_dim,
                embedding_dim=embedding_dim,
                random_dim=random_dim,
                generator_dims=generator_dims,
                discriminator_dims=discriminator_dims,
                compress_dims=compress_dims,
                decompress_dims=decompress_dims,
                bn_decay=bn_decay,
                l2_scale=l2_scale,
                data_type=data_type,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1,
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
        Generate data using the MedGAN model.

        Args:
            size (int): The number of samples to generate. If not specified, defaults to the training sample size.

        Returns:
            pd.DataFrame: The generated data.

        Raises:
            ValueError: If there is a value error such as invalid size.
            TypeError: If there is a type error such as wrong data type for size.
            AttributeError: If the model does not have a sample method.
            Exception: For any other exceptions that occur during data generation.
        """
        try:
            print("[INFO] Generating data using MedGAN model")
            if size is None:
                size = self.training_sample_size

            generated_data = self.model.sample(size, verbose=0)
            generated_data_decoded = self.postprocess_data(generated_data)
            print("[SUCCESS] Data generation completed")
            return generated_data_decoded
        except ValueError as e:
            print(f"[ERROR] ValueError during data generation: {e}")
        except TypeError as e:
            print(f"[ERROR] TypeError during data generation: {e}")
        except AttributeError as e:
            print(f"[ERROR] AttributeError during data generation: {e}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred during data generation: {e}")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import warnings

warnings.filterwarnings("ignore")


class MedGAN:
    """
    The MedGAN Model for generating synthetic tabular data.
    """

    def __init__(self) -> None:
        self.input_dim = None
        self.embedding_dim = None
        self.random_dim = None
        self.generator_dims = None
        self.discriminator_dims = None
        self.compress_dims = None
        self.decompress_dims = None
        self.bn_decay = None
        self.l2_scale = None
        self.data_type = None
        self.autoencoder = None
        self.generator = None
        self._ordinal_encoder = OrdinalEncoder(
            dtype=int, handle_unknown="use_encoded_value", unknown_value=-1
        )
        self._label_encoder = LabelEncoder()

    def fit(
        self,
        data,
        input_dim,
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
        batch_size=32,
        pretrain_epochs=1,
        verbose=1,
    ) -> None:
        """
        Fit the MedGAN model to the data.

        Parameters
        ----------
        data : np.ndarray
            The input data for training, in tabular format.

        input_dim : int
            The dimensionality of the input data.

        embedding_dim : int, default=128
            The dimensionality of the embedding vector.

        random_dim : int, default=128
            The dimensionality of the random noise vector.

        generator_dims : tuple, default=(128, 128)
            The dimensions of the generator layers.

        discriminator_dims : tuple, default=(256, 128, 1)
            The dimensions of the discriminator layers.

        compress_dims : tuple, default=()
            The dimensions of the compression layers in the autoencoder.

        decompress_dims : tuple, default=()
            The dimensions of the decompression layers in the autoencoder.

        bn_decay : float, default=0.99
            Decay rate for batch normalization.

        l2_scale : float, default=0.001
            L2 regularization scale.

        data_type : str, default='binary'
            Type of data ('binary' or 'count').

        batch_size : int, default=32
            Batch size for training.

        epochs : int, default=10
            Number of epochs for training.

        pretrain_epochs : int, default=1
            Number of epochs for pre-training the autoencoder.

        verbose : int, default=1
            Verbosity mode (0 = silent, 1 = progress bar).
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.generator_dims = list(generator_dims) + [embedding_dim]
        self.discriminator_dims = discriminator_dims
        self.compress_dims = list(compress_dims) + [embedding_dim]
        self.decompress_dims = list(decompress_dims) + [input_dim]
        self.bn_decay = bn_decay
        self.l2_scale = l2_scale
        self.data_type = data_type

        print(f"[INFO] Data shape: {data.shape}")
        data = self._ordinal_encoder.fit_transform(data)
        print(f"[INFO] Encoded data shape: {data.shape}")

        trainX, validX = train_test_split(data, test_size=0.1, random_state=0)
        print(f"[INFO] TrainX shape: {trainX.shape}, ValidX shape: {validX.shape}")

        if verbose:
            print("Pretraining autoencoder...")
        self._pretrain_autoencoder(trainX, pretrain_epochs, batch_size, verbose)

        if verbose:
            print("Training GAN...")
        self._train_gan(trainX, validX, epochs, batch_size, verbose)

    def _pretrain_autoencoder(self, trainX, pretrain_epochs, batch_size, verbose):
        """
        Pretrain the autoencoder part of MedGAN.
        """
        tf.keras.backend.clear_session()
        input_data = tf.keras.Input(shape=(self.input_dim,))
        x = input_data
        print(f"[INFO] Autoencoder input shape: {input_data.shape}")

        # Encoder
        for i, dim in enumerate(self.compress_dims[:-1]):
            x = tf.keras.layers.Dense(dim, activation=self._get_activation())(x)
            print(f"[INFO] After Encoder Dense layer {i}, shape: {x.shape}")

        encoded = tf.keras.layers.Dense(
            self.compress_dims[-1], activation=self._get_activation()
        )(x)
        print(f"[INFO] Encoded layer shape: {encoded.shape}")

        # Decoder
        x = encoded
        for i, dim in enumerate(self.decompress_dims[:-1]):
            x = tf.keras.layers.Dense(dim, activation=self._get_activation())(x)
            print(f"[INFO] After Decoder Dense layer {i}, shape: {x.shape}")

        decoded = tf.keras.layers.Dense(
            self.input_dim,
            activation="sigmoid" if self.data_type == "binary" else "relu",
        )(x)
        print(f"[INFO] Decoded output shape: {decoded.shape}")

        autoencoder = tf.keras.models.Model(inputs=input_data, outputs=decoded)
        autoencoder.compile(
            optimizer="adam",
            loss="binary_crossentropy" if self.data_type == "binary" else "mse",
        )

        autoencoder.fit(
            trainX,
            trainX,
            batch_size=batch_size,
            epochs=pretrain_epochs,
            verbose=verbose,
        )
        self.autoencoder = autoencoder
        tf.keras.backend.clear_session()

    def _train_gan(self, trainX, validX, epochs, batch_size, verbose):
        """
        Train the GAN component of MedGAN.
        """
        tf.keras.backend.clear_session()

        # Build the Generator
        self.generator = self._build_generator()
        print(f"[INFO] Generator model summary:")
        self.generator.summary()  # Ensure full summary is printed

        # Build the Discriminator
        discriminator = self._build_discriminator()
        discriminator.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        print(f"[INFO] Discriminator model summary:")
        discriminator.summary()  # Ensure full summary is printed

        # Build and compile the GAN
        random_input = tf.keras.Input(shape=(self.random_dim,))
        generated_data = self.generator(random_input)
        discriminator.trainable = False
        gan_output = discriminator(generated_data)
        gan = tf.keras.models.Model(inputs=random_input, outputs=gan_output)
        gan.compile(optimizer="adam", loss="binary_crossentropy")
        print(f"[INFO] GAN model summary:")
        gan.summary()  # Ensure full summary is printed

        for epoch in range(epochs):
            # Generate synthetic data
            random_noise = np.random.normal(size=(trainX.shape[0], self.random_dim))
            generated_data = self.generator.predict(random_noise)
            print(f"[INFO] Generated data shape: {generated_data.shape}")

            # Train Discriminator
            real_labels = np.ones((trainX.shape[0], 1))
            fake_labels = np.zeros((trainX.shape[0], 1))
            d_loss_real = discriminator.train_on_batch(trainX, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)

            # Train Generator
            misleading_labels = np.ones((trainX.shape[0], 1))
            g_loss = gan.train_on_batch(random_noise, misleading_labels)

            if verbose:
                try:
                    # Attempt to format the loss values correctly
                    d_loss_real_str = f"{d_loss_real[0]:.4f}"
                    d_loss_fake_str = f"{d_loss_fake[0]:.4f}"
                    g_loss_str = f"{g_loss:.4f}"
                    print(
                        f"Epoch {epoch+1}/{epochs}: D_loss_real = {d_loss_real_str}, D_loss_fake = {d_loss_fake_str}, G_loss = {g_loss_str}"
                    )
                except TypeError as e:
                    print(f"[ERROR] TypeError during logging: {e}")
                    print(
                        f"[DEBUG] d_loss_real: {d_loss_real}, d_loss_fake: {d_loss_fake}, g_loss: {g_loss}"
                    )

        tf.keras.backend.clear_session()

    def _build_generator(self):
        """
        Build the generator model.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.random_dim,)))
        print(f"[INFO] Generator InputLayer shape: {(self.random_dim,)}")
        for i, dim in enumerate(self.generator_dims[:-1]):
            model.add(tf.keras.layers.Dense(dim, activation="relu"))
            print(f"[INFO] After Generator Dense layer {i}, shape: {dim}")
        model.add(
            tf.keras.layers.Dense(
                self.input_dim,
                activation="tanh" if self.data_type == "binary" else "relu",
            )
        )
        print(f"[INFO] Generator output shape: {self.input_dim}")
        return model

    def _build_discriminator(self):
        """
        Build the discriminator model.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.input_dim,)))
        print(f"[INFO] Discriminator InputLayer shape: {(self.input_dim,)}")
        for i, dim in enumerate(self.discriminator_dims[:-1]):
            model.add(tf.keras.layers.Dense(dim, activation="relu"))
            print(f"[INFO] After Discriminator Dense layer {i}, shape: {dim}")
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        print(f"[INFO] Discriminator output shape: 1")
        return model

    def _generate_samples(self, n_samples, verbose):
        """
        Generate synthetic samples using the trained generator.
        """
        random_noise = np.random.normal(size=(n_samples, self.random_dim))
        synthetic_data = self.generator.predict(random_noise, verbose=verbose)
        print(f"[INFO] Generated samples shape: {synthetic_data.shape}")
        return synthetic_data

    def _get_activation(self):
        """
        Get the appropriate activation function for the autoencoder.
        """
        return tf.nn.tanh if self.data_type == "binary" else tf.nn.relu

    def sample(self, n_samples=100, verbose=0) -> np.ndarray:
        """
        Generate synthetic data using the trained generator.

        Parameters
        ----------
        n_samples : int, default=100
            Number of synthetic samples to generate.

        verbose : int, default=0
            Verbosity mode.

        Returns
        -------
        np.ndarray
            Generated synthetic data.
        """
        if not hasattr(self, "generator"):
            raise AttributeError(
                "The generator model must be trained before sampling data."
            )

        # Generate synthetic samples
        random_noise = np.random.normal(size=(n_samples, self.random_dim))
        synthetic_data = self.generator.predict(random_noise, verbose=verbose)
        print(f"[INFO] Generated samples shape: {synthetic_data.shape}")
        return synthetic_data

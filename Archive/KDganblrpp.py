import numpy as np
from tensorflow.keras import layers, models
import logging

# Set up logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GANBLRPlusPlus:
    def __init__(self, input_dim, latent_dim):
        # Input validation
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer.")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        """Builds the generator model."""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.latent_dim),
            layers.Dense(self.input_dim, activation='sigmoid')
        ])
        return model

    def build_discriminator(self):
        """Builds the discriminator model."""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.input_dim),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, data, epochs=1000, batch_size=32):
        """Trains the GANBLR++ model."""
        for epoch in range(epochs):
            # Sample random batch of real data
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]
            real_labels = np.ones((batch_size, 1))

            # Generate fake data
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict(noise)
            fake_labels = np.zeros((batch_size, 1))

            # Train discriminator on both real and fake data
            real_loss = self.discriminator.train_on_batch(real_data, real_labels)
            fake_loss = self.discriminator.train_on_batch(fake_data, fake_labels)
            d_loss = 0.5 * np.add(real_loss, fake_loss)

            # Train generator to fool the discriminator
            misleading_labels = np.ones((batch_size, 1))
            g_loss = self.discriminator.train_on_batch(fake_data, misleading_labels)

            # Log progress every 100 epochs
            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}/{epochs}: [D loss: {d_loss}] [G loss: {g_loss}]")

    def save_models(self, generator_path="generator.h5", discriminator_path="discriminator.h5"):
        """Saves the generator and discriminator models."""
        self.generator.save(generator_path)
        self.discriminator.save(discriminator_path)

    def load_models(self, generator_path="generator.h5", discriminator_path="discriminator.h5"):
        """Loads the generator and discriminator models."""
        self.generator = models.load_model(generator_path)
        self.discriminator = models.load_model(discriminator_path)

    def generate_batch(self, batch_size):
        """Generates a batch of synthetic data."""
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        return self.generator.predict(noise)

def generate_synthetic_data(data=None, input_dim=10, latent_dim=20):
    """Generates synthetic data using the GANBLR++ model."""
    if data is None:
        data = np.random.rand(1000, input_dim)  # Default synthetic data

    gan = GANBLRPlusPlus(input_dim=input_dim, latent_dim=latent_dim)
    gan.train(data, epochs=1000, batch_size=64)

    # Generate new synthetic data
    noise = np.random.normal(0, 1, (100, latent_dim))
    synthetic_data = gan.generator.predict(noise)
    return synthetic_data

if __name__ == "__main__":
    # Example usage
    real_data = np.random.rand(1000, 10)  # Replace with actual data as needed
    synthetic_data = generate_synthetic_data(real_data)

    # Print the shape of generated data
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")
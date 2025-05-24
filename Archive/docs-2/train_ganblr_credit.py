import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

# Load preprocessed training data
X_train = pd.read_csv('credit_X_train.csv').to_numpy()

# GAN parameters
input_dim = X_train.shape[1]
latent_dim = 100
batch_size = 256
epochs = 5000

# Build the Generator
generator = Sequential([
    Dense(128, input_dim=latent_dim),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),
    Dense(256),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),
    Dense(input_dim, activation='tanh')
])

# Build the Discriminator
discriminator = Sequential([
    Dense(256, input_dim=input_dim),
    LeakyReLU(alpha=0.2),
    Dense(128),
    LeakyReLU(alpha=0.2),
    Dense(1, activation='sigmoid')
])

# Compile the Discriminator
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Build and compile the GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Training the GAN
real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in range(epochs):
    try:
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict(noise)

        # Train on real and fake data
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Log progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")

    except Exception as e:
        print(f"Error at epoch {epoch}: {e}")
        break

# Generate synthetic data
noise = np.random.normal(0, 1, (1000, latent_dim))
synthetic_data = generator.predict(noise)

# Save synthetic data
synthetic_df = pd.DataFrame(synthetic_data, columns=[f"Feature_{i}" for i in range(input_dim)])
synthetic_df.to_csv('credit_synthetic_data.csv', index=False)

print("Training complete. Synthetic data saved as 'credit_synthetic_data.csv'.")

import pandas as pd
import numpy as np
import warnings
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization
from keras.optimizers import Adam

# Suppress TensorFlow/Keras warnings for clean output
warnings.filterwarnings('ignore')

# Load preprocessed training data
try:
    X_train = pd.read_csv('credit_X_train.csv').to_numpy()
    print("Training data loaded successfully!")
except Exception as e:
    print(f"Error loading training data: {e}")
    exit(1)

# GAN parameters
input_dim = X_train.shape[1]  # Number of features
latent_dim = 100  # Noise vector size
batch_size = 256
epochs = 5000
save_interval = 1000  # Save data periodically

# Build the Generator
generator = Sequential([
    Dense(128, input_dim=latent_dim),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),
    Dense(256),
    LeakyReLU(alpha=0.2),
    BatchNormalization(momentum=0.8),
    Dense(input_dim, activation='tanh')  # Generate data matching input dimension
])

# Build the Discriminator
discriminator = Sequential([
    Dense(256, input_dim=input_dim),
    LeakyReLU(alpha=0.2),
    Dense(128),
    LeakyReLU(alpha=0.2),
    Dense(1, activation='sigmoid')  # Output probability of real/fake
])

# Compile the Discriminator
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# Build and compile the GAN
discriminator.trainable = False  # Freeze discriminator during generator training
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Display Model Summaries
print("Generator Summary:")
generator.summary()
print("\nDiscriminator Summary:")
discriminator.summary()

# Labels for real and fake data
real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

# Training the GAN
print("\nStarting GAN training...")
for epoch in range(epochs):
    try:
        # Train Discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        generated_data = generator.predict_on_batch(noise)

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Log progress every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")

        # Save synthetic data periodically
        if epoch % save_interval == 0 and epoch != 0:
            noise = np.random.normal(0, 1, (1000, latent_dim))
            synthetic_data = generator.predict_on_batch(noise)
            synthetic_df = pd.DataFrame(synthetic_data, columns=[f"Feature_{i}" for i in range(input_dim)])
            synthetic_df.to_csv(f'credit_synthetic_data_epoch_{epoch}.csv', index=False)
            print(f"Synthetic data saved at epoch {epoch}.")

    except Exception as e:
        print(f"Error at epoch {epoch}: {e}")
        break

# Final synthetic data generation
print("\nGenerating final synthetic data...")
noise = np.random.normal(0, 1, (1000, latent_dim))
synthetic_data = generator.predict_on_batch(noise)
synthetic_df = pd.DataFrame(synthetic_data, columns=[f"Feature_{i}" for i in range(input_dim)])
synthetic_df.to_csv('credit_synthetic_data.csv', index=False)

print("Training complete. Final synthetic data saved as 'credit_synthetic_data.csv'.")

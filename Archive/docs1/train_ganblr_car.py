import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the encoded dataset
file_path = "car_dataset_encoded.csv"
df = pd.read_csv(file_path)

# Split the dataset into features (X) and target (y)
X = df.drop("Class", axis=1)  # Features
y = df["Class"]  # Target

# Normalize features for GAN training
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# GAN Model Definitions
def build_generator(input_dim, output_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(output_dim, activation="tanh")
    ])
    return model

def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model

# Define GAN components
input_dim = X_train.shape[1]
generator = build_generator(input_dim=input_dim, output_dim=input_dim)
discriminator = build_discriminator(input_dim=input_dim)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

# Loss Function
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

# GAN Training
epochs = 10000
batch_size = 32

for epoch in range(epochs):
    # Generate fake data
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    fake_data = generator(noise, training=True)

    # Sample real data
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_data = X_train[idx]

    # Train discriminator
    with tf.GradientTape() as tape:
        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(fake_data, training=True)

        d_loss_real = binary_cross_entropy(tf.ones_like(real_output), real_output)
        d_loss_fake = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
        d_loss = (d_loss_real + d_loss_fake) / 2

    gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Train generator
    with tf.GradientTape() as tape:
        fake_data = generator(noise, training=True)
        fake_output = discriminator(fake_data, training=True)
        g_loss = binary_cross_entropy(tf.ones_like(fake_output), fake_output)

    gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # Print losses every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Discriminator Loss: {d_loss.numpy()}, Generator Loss: {g_loss.numpy()}")

# Generate synthetic data
synthetic_data = generator.predict(np.random.normal(0, 1, (X_test.shape[0], input_dim)))

# Rescale synthetic data back to original scale
synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

# Save synthetic data
synthetic_df = pd.DataFrame(synthetic_data_rescaled, columns=X.columns)
synthetic_df.to_csv("car_synthetic_data.csv", index=False)
print("\nSynthetic data saved as 'car_synthetic_data.csv'.")

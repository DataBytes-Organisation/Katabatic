import tensorflow as tf
import numpy as np

class Medgan(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim=128, random_dim=128, ae_loss_type='mse'):
        super(Medgan, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.random_dim = random_dim
        self.ae_loss_type = ae_loss_type

        # Autoencoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

        # Generator
        self.generator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(random_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

        # Discriminator (No sigmoid at the end for logits output)
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def compute_ae_loss(self, real_data):
        reconstructed = self.decoder(self.encoder(real_data))
        if self.ae_loss_type == 'bce':
            return tf.reduce_mean(tf.keras.losses.binary_crossentropy(real_data, reconstructed))
        else:
            return tf.reduce_mean(tf.math.squared_difference(real_data, reconstructed))

    def train_step(self, real_data, random_data):
        generated_data = self.generator(random_data)

        # Autoencoder loss
        ae_loss = self.compute_ae_loss(real_data)

        # Discriminator loss (use logits)
        real_logits = self.discriminator(real_data)
        fake_logits = self.discriminator(generated_data)
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_logits), logits=real_logits))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_logits), logits=fake_logits))
        d_loss = d_loss_real + d_loss_fake

        # Generator loss
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_logits), logits=fake_logits))

        return ae_loss, d_loss, g_loss

    def generate_data(self, num_samples=100):
        random_data = np.random.normal(size=(num_samples, self.random_dim))
        generated_data = self.generator(random_data)
        return generated_data.numpy()





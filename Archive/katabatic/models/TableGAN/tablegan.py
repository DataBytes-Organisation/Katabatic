import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class TableGAN:
    def __init__(self, input_dim, label_dim, z_dim=100, delta_mean=0.1, delta_sd=0.1):
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.z_dim = z_dim
        self.delta_mean = delta_mean
        self.delta_sd = delta_sd
        
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        self.classifier = self._build_classifier()
        
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.c_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        
        self.f_X_mean = None
        self.f_X_sd = None
        self.f_Z_mean = None
        self.f_Z_sd = None

    def _build_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(256 * 4 * 4, input_shape=(self.z_dim,), use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Reshape((4, 4, 256)),
            layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Conv2DTranspose(1, 4, strides=2, padding='same', use_bias=False, activation='tanh')
        ])
        return model

    def _build_discriminator(self):
        inputs = layers.Input(shape=(32, 32, 1))
        x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        features = layers.Flatten()(x)
        output = layers.Dense(1)(features)
        
        return tf.keras.Model(inputs, [output, features])

    def _build_classifier(self):
        model = tf.keras.Sequential([
            layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(32, 32, 1)),
            layers.LeakyReLU(0.2),
            layers.Conv2D(128, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Conv2D(256, 4, strides=2, padding='same'),
            layers.LeakyReLU(0.2),
            layers.Flatten(),
            layers.Dense(self.label_dim, activation='softmax')
        ])
        return model

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    def gradient_penalty(self, real, fake):
        alpha = tf.random.uniform([tf.shape(real)[0], 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        interpolated = real + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred, _ = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_data, real_labels):
        real_data = tf.cast(real_data, tf.float32)
        real_labels = tf.cast(real_labels, tf.float32)
        batch_size = tf.shape(real_data)[0]
        noise = tf.random.normal([batch_size, self.z_dim])

        with tf.GradientTape() as d_tape, tf.GradientTape() as c_tape, tf.GradientTape() as g_tape:
            fake_data = self.generator(noise, training=True)
            
            real_output, real_features = self.discriminator(real_data, training=True)
            fake_output, fake_features = self.discriminator(fake_data, training=True)
            
            d_loss = self.wasserstein_loss(real_output, -tf.ones_like(real_output)) + \
                     self.wasserstein_loss(fake_output, tf.ones_like(fake_output))
            gp = self.gradient_penalty(real_data, fake_data)
            d_loss += 10.0 * gp
            
            f_X_mean = tf.reduce_mean(real_features, axis=0)
            f_X_sd = tf.math.reduce_std(real_features, axis=0)
            f_Z_mean = tf.reduce_mean(fake_features, axis=0)
            f_Z_sd = tf.math.reduce_std(fake_features, axis=0)
            
            if self.f_X_mean is None:
                self.f_X_mean = tf.Variable(tf.zeros_like(f_X_mean), trainable=False)
                self.f_X_sd = tf.Variable(tf.zeros_like(f_X_sd), trainable=False)
                self.f_Z_mean = tf.Variable(tf.zeros_like(f_Z_mean), trainable=False)
                self.f_Z_sd = tf.Variable(tf.zeros_like(f_Z_sd), trainable=False)
            
            self.f_X_mean.assign(0.99 * self.f_X_mean + 0.01 * f_X_mean)
            self.f_X_sd.assign(0.99 * self.f_X_sd + 0.01 * f_X_sd)
            self.f_Z_mean.assign(0.99 * self.f_Z_mean + 0.01 * f_Z_mean)
            self.f_Z_sd.assign(0.99 * self.f_Z_sd + 0.01 * f_Z_sd)
            
            L_mean = tf.reduce_sum(tf.square(self.f_X_mean - self.f_Z_mean))
            L_sd = tf.reduce_sum(tf.square(self.f_X_sd - self.f_Z_sd))
            
            g_info_loss = tf.maximum(0.0, L_mean - self.delta_mean) + tf.maximum(0.0, L_sd - self.delta_sd)
            
            c_real_pred = self.classifier(real_data, training=True)
            c_fake_pred = self.classifier(fake_data, training=True)
            
            c_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(real_labels, c_real_pred))
            g_class_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(real_labels, c_fake_pred))
            
            g_loss = -tf.reduce_mean(fake_output) + 0.1 * g_info_loss + 0.1 * g_class_loss

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        c_gradients = c_tape.gradient(c_loss, self.classifier.trainable_variables)
        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        self.c_optimizer.apply_gradients(zip(c_gradients, self.classifier.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return d_loss, g_loss, c_loss

    def fit(self, x, y, batch_size=64, epochs=100, verbose=1):
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size)

        for epoch in range(epochs):
            d_losses, g_losses, c_losses = [], [], []
            for batch_x, batch_y in dataset:
                d_loss, g_loss, c_loss = self.train_step(batch_x, batch_y)
                d_losses.append(d_loss)
                g_losses.append(g_loss)
                c_losses.append(c_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: [D loss: {np.mean(d_losses):.4f}] [G loss: {np.mean(g_losses):.4f}] [C loss: {np.mean(c_losses):.4f}]")

        return self

    def sample(self, n_samples):
        noise = tf.random.normal([n_samples, self.z_dim])
        generated_data = self.generator(noise, training=False)
        generated_labels = self.classifier(generated_data, training=False)
        return generated_data.numpy(), generated_labels.numpy()
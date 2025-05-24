
import numpy as np
import pickle
import os
from medgan import Medgan  # assumes medgan.py is in the same directory
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class MedGANInterface:
    def __init__(self, matrix_path, model_path='medgan_model.ckpt-999', z_dim=128):
        self.matrix_path = matrix_path
        self.model_path = model_path
        self.z_dim = z_dim
        self.data = None
        self.model = None
        self.sess = tf.Session()
        self._load_matrix()
        self._load_model()

    def _load_matrix(self):
        # Load the matrix file
        self.data = np.load(self.matrix_path, allow_pickle=True)

        # ⚠️ Force input dimension to match trained model
        self.data_dim = 295
    def _load_model(self):
        # Create MedGAN model instance
        self.model = Medgan(inputDim=295)

        # Create placeholders
        self.x_random = tf.placeholder('float', [None, self.z_dim])
        self.bn_train = tf.placeholder(tf.bool)

        # Build graph: autoencoder + generator
        x_dummy = tf.placeholder('float', [None, self.data_dim])
        _, decodeVariables = self.model.buildAutoencoder(x_dummy)
        x_emb = self.model.buildGeneratorTest(self.x_random, self.bn_train)

        # Decode generator output
        tempVec = x_emb
        for i, _ in enumerate(self.model.decompressDims[:-1]):
            tempVec = self.model.aeActivation(
                tf.add(tf.matmul(tempVec, decodeVariables[f'aed_W_{i}']), decodeVariables[f'aed_b_{i}'])
            )
        final_i = len(self.model.decompressDims) - 1
        if self.model.dataType == 'binary':
            self.fake_data = tf.nn.sigmoid(tf.add(
                tf.matmul(tempVec, decodeVariables[f'aed_W_{final_i}']),
                decodeVariables[f'aed_b_{final_i}']
            ))
        else:
            self.fake_data = tf.nn.relu(tf.add(
                tf.matmul(tempVec, decodeVariables[f'aed_W_{final_i}']),
                decodeVariables[f'aed_b_{final_i}']
            ))

        # Now load the model
        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)






    def generate(self, num_samples=100):
        random_z = np.random.normal(size=(num_samples, self.z_dim))
        samples = self.sess.run(self.fake_data, feed_dict={
            self.x_random: random_z,
            self.bn_train: False
        })
        return samples


import tensorflow as tf
import numpy as np

class DummyLayer(tf.keras.layers.Layer):

    def call(self, inputs):

        return tf.nn.softmax(np.random.rand(inputs.shape[0], 10))

    # Add the custom latent space method
    def custom_latent_space(self, inputs):
        """
        This method returns a custom latent space representation.
        """
        return np.random.random((inputs.shape[0], 512))

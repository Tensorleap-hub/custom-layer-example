import tensorflow as tf

from custom_layers.mnist_custom_layer import TorchResNetLayer

# Define the input shape
input_shape = (28, 28, 1)  # Example shape for an RGB image

# Create the input layer
inputs = tf.keras.Input(shape=input_shape)

# Identity layer that outputs the input as is
identity = tf.keras.layers.Lambda(lambda x: x)(inputs)

# custom_layer = TorchResNetLayer()(identity)

# Create the model
model = tf.keras.Model(inputs=inputs, outputs=identity)
model.summary()
model.save('model/model.h5')

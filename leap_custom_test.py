import tensorflow as tf
from custom_layers.mnist_custom_layer import TorchMNISTLayer
from leap_binder import gt_encoder, input_encoder, preprocess_func


def check_integration():
    responses = preprocess_func()
    train = responses[0]
    idx = 0
    image = input_encoder(idx, train)
    gt = gt_encoder(idx, train)

    model = tf.keras.models.load_model('model/model.h5')

    custom_layer = TorchMNISTLayer()

    model_output = model(image[tf.newaxis, ...])
    custom_layer_output = custom_layer(image[tf.newaxis, ...])
    custom_layer_lt = custom_layer.custom_latent_space(image[tf.newaxis, ...])

    print('Done')


if __name__ == '__main__':
    check_integration()
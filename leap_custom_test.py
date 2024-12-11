import tensorflow as tf
import numpy as np

from custom_layers.branched_custom_layer import BranchLayer
from custom_layers.input_custom_layer import InputLayer
from custom_layers.mnist_custom_layer import TorchMNISTLayer
from leap_binder import (
    dummy_input_encoder,
    dummy_loss,
    gt_encoder,
    input_encoder,
    branch_input_0_encoder,
    branch_input_1_encoder,
    preprocess_func,
)


def check_integration():
    responses = preprocess_func()
    train = responses[0]
    idx = 0
    image = input_encoder(idx, train)
    dummy_input = dummy_input_encoder(idx, train)
    branch_input_0 = branch_input_0_encoder(idx, train)
    branch_input_1 = branch_input_1_encoder(idx, train)
    gt = gt_encoder(idx, train)

    model = tf.keras.models.load_model("model/model.h5")

    custom_pred_layer = TorchMNISTLayer()
    custom_input_layer = InputLayer()
    custom_branch_layer = BranchLayer()

    custom_input_layer_output = custom_input_layer(
        [dummy_input, image[tf.newaxis, ...]]
    )

    model_output = model(custom_input_layer_output)

    custom_pred_layer_output = custom_pred_layer(model_output)
    custom_pred_layer_lt = custom_pred_layer.custom_latent_space(model_output)

    custom_branch_layer_output = custom_branch_layer(
        [branch_input_0[tf.newaxis, ...], branch_input_1[tf.newaxis, ...]]
    )

    loss = dummy_loss(
        tf.convert_to_tensor(np.expand_dims(gt, 0)),
        custom_pred_layer_output,
        custom_branch_layer_output,
    )

    print("Done")


if __name__ == "__main__":
    check_integration()

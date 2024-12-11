import os
from typing import List, Union

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import Metric, DatasetMetadataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
import torch

from config import CONFIG
from custom_layers.branched_custom_layer import BranchLayer
from custom_layers.input_custom_layer import InputLayer
from custom_layers.mnist_custom_layer import TorchMNISTLayer
from model.mnist_torch_model import MNISTPyTorchModel
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_input_encoder,
    tensorleap_gt_encoder,
    tensorleap_metadata,
    tensorleap_custom_loss,
)


@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    # (train_X, train_Y), (val_X, val_Y) = mnist.load_data()
    (train_X, train_Y), (val_X, val_Y) = (
        np.random.rand(60000, 28, 28),
        np.random.randint(0, 10, 60000),
    ), (np.random.rand(10000, 28, 28), np.random.randint(0, 10, 10000))
    train_X = np.expand_dims(train_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    train_X = train_X / 255  # Normalize to [0,1]
    train_Y = to_categorical(train_Y)  # Hot Vector

    val_X = np.expand_dims(val_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    val_X = val_X / 255  # Normalize to [0,1]
    val_Y = to_categorical(val_Y)  # Hot Vector

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs
    train = PreprocessResponse(
        length=len(train_X), data={"images": train_X, "labels": train_Y}
    )
    val = PreprocessResponse(length=len(val_X), data={"images": val_X, "labels": val_Y})
    response = [train, val]
    return response


@tensorleap_input_encoder("image")
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data["images"][idx].astype("float32")


@tensorleap_input_encoder("dummy_input")
def dummy_input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return np.random.rand(1, 8, 16, 16, 1).astype("float32")


@tensorleap_input_encoder("branch_input_0")
def branch_input_0_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data["images"][idx].astype("float32")


@tensorleap_input_encoder("branch_input_1")
def branch_input_1_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data["images"][idx].astype("float32")


@tensorleap_gt_encoder("classes")
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data["labels"][idx].astype("float32")


@tensorleap_metadata("digit")
def metadata(idx: int, preprocess: PreprocessResponse) -> Union[str, int, float]:
    return np.argmax(preprocess.data["labels"][idx])


@tensorleap_custom_loss("cce")
def dummy_loss(y_true, y_pred, branch_pred):
    print(f"y_true: {y_true.shape}")
    print(f"y_pred n outputs: {len(y_pred)}")
    print(f"y_pred 0: {y_pred[0].shape}")
    print(f"y_pred 1: {y_pred[1].shape}")
    print(f"branch_pred: {branch_pred.shape}")
    y_pred = y_pred[0]
    loss = CategoricalCrossentropy()
    return loss(y_true, y_pred)


leap_binder.set_custom_layer(
    custom_layer=TorchMNISTLayer, name="TorchMNISTLayer", use_custom_latent_space=True
)
leap_binder.set_custom_layer(
    custom_layer=BranchLayer, name="BranchLayer", use_custom_latent_space=False
)
leap_binder.set_custom_layer(
    custom_layer=InputLayer, name="InputLayer", use_custom_latent_space=False
)

if __name__ == "__main__":
    leap_binder.check()

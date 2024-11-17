import os
from typing import List, Union

import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

# Tensorleap imports
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse 
from code_loader.contract.enums import Metric, DatasetMetadataType
from code_loader.contract.visualizer_classes import LeapHorizontalBar
import torch

from config import CONFIG
from custom_layers.mnist_custom_layer import TorchMNISTLayer
from model.mnist_torch_model import MNISTPyTorchModel
from code_loader.inner_leap_binder.leapbinder_decorators import *


@tensorleap_preprocess()
def preprocess_func() -> List[PreprocessResponse]:
    (train_X, train_Y), (val_X, val_Y) = mnist.load_data()

    train_X = np.expand_dims(train_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    train_X = train_X / 255                       # Normalize to [0,1]
    train_Y = to_categorical(train_Y)           # Hot Vector
    
    val_X = np.expand_dims(val_X, axis=-1)  # Reshape :,28,28 -> :,28,28,1
    val_X = val_X / 255                     # Normalize to [0,1]
    val_Y = to_categorical(val_Y)           # Hot Vector

    # Generate a PreprocessResponse for each data slice, to later be read by the encoders.
    # The length of each data slice is provided, along with the data dictionary.
    # In this example we pass `images` and `labels` that later are encoded into the inputs and outputs 
    train = PreprocessResponse(length=len(train_X), data={'images': train_X, 'labels': train_Y})
    val = PreprocessResponse(length=len(val_X), data={'images': val_X, 'labels': val_Y})
    response = [train, val]
    return response

@tensorleap_input_encoder('image')
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['images'][idx].astype('float32')

@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['labels'][idx].astype('float32')

@tensorleap_metadata('digit')
def metadata(idx: int, preprocess: PreprocessResponse) -> Union[str, int, float]:
    return np.argmax(preprocess.data['labels'][idx])



leap_binder.set_custom_layer(custom_layer=TorchMNISTLayer, name='TorchMNISTLayer', use_custom_latent_space=True)

if __name__ == '__main__':
    leap_binder.check()

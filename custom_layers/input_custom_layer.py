import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
from model.mnist_torch_model import MNISTPyTorchModel
from config import CONFIG


class InputLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(InputLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model = MNISTPyTorchModel()
        self.torch_model.load_state_dict(
            torch.load(CONFIG["model_path"], map_location="cpu")
        )
        self.torch_model.eval()

    def call(self, inputs):
        print(f"Input 0 shape: {inputs[0].shape}")
        print(f"Input 0 type: {type(inputs[0])}")
        print(f"Input 1 shape: {inputs[1].shape}")
        print(f"Input 1 type: {type(inputs[1])}")
        inputs_pt = torch.from_numpy(inputs[1].numpy().transpose((0, 3, 1, 2)))
        print("inputs parsed")
        outputs = self.run_pytorch_model(inputs_pt)
        print("outputs generated")
        outputs = tf.nn.softmax(outputs)
        dummy_output = tf.convert_to_tensor(np.random.rand(1, 28, 28, 1))
        print("input custom layer done\n")
        return dummy_output

    def custom_latent_space(self, inputs):
        latent = self.run_pytorch_model(inputs, embedding=True)
        return latent

    def run_pytorch_model(self, inputs, embedding=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model.to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.torch_model(inputs, embedding=embedding)
        outputs = outputs.cpu().numpy().astype(np.float32)
        return outputs

import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
from model.mnist_torch_model import MNISTPyTorchModel
from config import CONFIG


class BranchLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(BranchLayer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model = MNISTPyTorchModel()
        self.torch_model.load_state_dict(
            torch.load(CONFIG["model_path"], map_location="cpu")
        )
        self.torch_model.eval()

    def call(self, inputs):
        print(f"Input shape: {inputs[0].shape}")
        print(f"Input type: {type(inputs[0])}")
        inputs = [
            torch.from_numpy(inp.numpy().transpose((0, 3, 1, 2))) for inp in inputs
        ]
        print("inputs parsed")
        outputs = self.run_pytorch_model(inputs[0])
        print(f"outputs shape: {outputs.shape}")
        res = tf.convert_to_tensor(np.random.rand(1, 1024))
        print(f"branch custom layer output shape: {res.shape}")
        print("branch custom layer done\n")
        return res

    def custom_latent_space(self, inputs):
        latent = self.run_pytorch_model(inputs[0], embedding=True)
        return latent

    def run_pytorch_model(self, inputs, embedding=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_model.to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.torch_model(inputs, embedding=embedding)
        outputs = outputs.cpu().numpy().astype(np.float32)
        return outputs

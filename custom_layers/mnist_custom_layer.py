import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
from model.mnist_torch_model import MNISTPyTorchModel
from config import CONFIG


class TorchMNISTLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TorchMNISTLayer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_model = MNISTPyTorchModel()
        self.torch_model.load_state_dict(torch.load(CONFIG['model_path'], map_location='cpu'))
        self.torch_model.eval()

    def call(self, inputs):
        outputs = self.run_pytorch_model(inputs)
        outputs = tf.nn.softmax(outputs)
        return outputs

    def run_pytorch_model(self, inputs, embedding=False):
        inputs = inputs.numpy()
        inputs = np.transpose(inputs, (0, 3, 1, 2))
        inputs = torch.from_numpy(inputs).float()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.torch_model.to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = self.torch_model(inputs, embedding=embedding)
        outputs = outputs.cpu().numpy().astype(np.float32)
        return outputs

    def custom_latent_space(self, inputs):
        latent = self.run_pytorch_model(inputs, embedding=True)
        return latent

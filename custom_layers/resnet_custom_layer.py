import tensorflow as tf
import torch
import torchvision.models as models
import numpy as np

class TorchResNetLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(TorchResNetLayer, self).__init__()
        # Initialize the PyTorch ResNet model
        self.torch_model = models.resnet18(pretrained=True)
        self.torch_model.eval()  # Set to evaluation mode

    def call(self, inputs):
        # Wrap the computation in a tf.py_function
        outputs = self.run_pytorch_model(inputs)
        # Set the shape of the outputs
        outputs.set_shape([None, 1000])  # ResNet outputs 1000 classes
        return outputs

    def run_pytorch_model(self, inputs):
        # Convert TensorFlow tensor to NumPy array
        inputs = inputs.numpy()

        # Rearrange dimensions from TensorFlow (batch, height, width, channels)
        # to PyTorch (batch, channels, height, width)
        inputs = np.transpose(inputs, (0, 3, 1, 2))

        # Convert NumPy array to PyTorch tensor
        inputs = torch.from_numpy(inputs).float()

        # Normalize the inputs as per ImageNet standards
        inputs = inputs / 255.0  # Assuming inputs are in [0, 255]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std

        # Move to the appropriate device
        self.torch_model.to(device)
        inputs = inputs.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = self.torch_model(inputs)

        # Convert PyTorch tensor back to NumPy array
        outputs = outputs.cpu().numpy().astype(np.float32)

        return outputs

    # Add the custom latent space method
    def custom_latent_space(self, inputs):
        """
        This method returns a custom latent space representation.
        """
        # Convert TensorFlow tensor to NumPy array
        inputs = inputs.numpy()
        # Rearrange dimensions for PyTorch
        inputs = np.transpose(inputs, (0, 3, 1, 2))
        # Convert NumPy array to PyTorch tensor
        inputs = torch.from_numpy(inputs).float()
        # Normalize inputs
        inputs = inputs / 255.0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        inputs = (inputs - mean) / std
        # Move inputs and model to device
        inputs = inputs.to(device)
        self.torch_model.to(device)
        # Extract features from the model
        features = torch.nn.Sequential(*list(self.torch_model.children())[:-1])
        with torch.no_grad():
            latent = features(inputs)
        # Flatten and convert to NumPy array
        latent = latent.view(latent.size(0), -1)
        latent = latent.cpu().numpy().astype(np.float32)
        return latent

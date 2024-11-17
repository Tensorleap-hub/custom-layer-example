import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTPyTorchModel(nn.Module):
    def __init__(self):
        super(MNISTPyTorchModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, embedding = False):
        x = F.relu(self.conv1(x))     
        x = F.relu(self.conv2(x))     
        x = F.max_pool2d(x, 2)        
        x = self.dropout1(x)
        x = torch.flatten(x, 1)       
        x = F.relu(self.fc1(x))  
        if embedding:
            return x     
        x = self.fc2(x)               
        return x

import torch.nn as nn
from ..utils.training_utils import calculate_fc1_input_size
from ..configs.model_config import config, input_dim, num_classes




class CNN(nn.Module):
    def __init__(self, num_classes=num_classes, num_fc_inputs = calculate_fc1_input_size(input_dim, config)):
        super(CNN, self).__init__()
        
        self.model = nn.Sequential(
            # Convolutional layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional layer block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional layer block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Flattening the layers
            nn.Flatten(),
            
            # Fully connected layers
            nn.Linear(num_fc_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
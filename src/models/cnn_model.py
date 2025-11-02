import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """
    CNN-based model for spatial pattern detection in network traffic.
    """
    
    def __init__(self, input_size: int = 41, num_classes: int = 5):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * (input_size // 8), 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout and activation
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape for CNN: (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
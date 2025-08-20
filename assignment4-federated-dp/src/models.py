"""
Neural network models for federated learning.
"""

import torch
import torch.nn as nn


class FederatedNet(nn.Module):
    """Simple neural network for breast cancer classification."""
    
    def __init__(self, input_size, hidden_size=64):
        super(FederatedNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x


def create_model(input_size, hidden_size=64):
    """Factory function to create a new model instance."""
    return FederatedNet(input_size, hidden_size)


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size):
    """Print a summary of the model architecture."""
    print(f"Model: {model.__class__.__name__}")
    print(f"Input size: {input_size}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Architecture:\n{model}")
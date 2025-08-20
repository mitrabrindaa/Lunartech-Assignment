"""
Flower client implementation for federated learning without differential privacy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning."""
    
    def __init__(self, hospital_data, model, device):
        self.hospital_data = hospital_data
        self.model = model
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Prepare data
        self.X = torch.FloatTensor(hospital_data.iloc[:, :-1].values).to(device)
        self.y = torch.FloatTensor(hospital_data.iloc[:, -1].values).unsqueeze(1).to(device)
        
        # Create train/val split
        train_size = int(0.8 * len(self.X))
        indices = torch.randperm(len(self.X))
        
        self.X_train = self.X[indices[:train_size]]
        self.y_train = self.y[indices[:train_size]]
        self.X_val = self.X[indices[train_size:]]
        self.y_val = self.y[indices[train_size:]]
        
        print(f"Client initialized with {len(self.X)} samples")
        print(f"  Train: {len(self.X_train)}, Val: {len(self.X_val)}")
        
    def get_parameters(self, config):
        """Return model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model."""
        self.set_parameters(parameters)
        self.model.train()
        
        epochs = config.get("epochs", 5)
        batch_size = config.get("batch_size", 32)
        
        # Create DataLoader
        dataset = TensorDataset(self.X_train, self.y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        return self.get_parameters(config={}), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Evaluate the model."""
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(self.X_val)
            loss = self.criterion(outputs, self.y_val)
            
            # Calculate accuracy
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == self.y_val).float().mean()
        
        return float(loss), len(self.X_val), {"accuracy": float(accuracy)}
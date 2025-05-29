import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
import torch.nn.functional as F
import numpy as np


class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # this is where i create the model itself
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)             # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, global_control: List[torch.tensor], local_control: List[torch.tensor]) -> tuple[float, float]:
        self.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            for idx, param in enumerate(self.parameters()):
                if param.grad is not None:
                    param.grad += self.c[idx] - self.ck[idx]  # Adjust gradient
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            batch_count += 1

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy, batch_count

    def test_epoch(self, test_loader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device) -> tuple[float, float]:
        self.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def get_model_parameters(self) -> List[np.ndarray]:
        return [param.detach().cpu().numpy() for param in self.state_dict().values()]

    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        state_dict = self.state_dict()
        for key, param_array in zip(state_dict.keys(), parameters):
            param_tensor = torch.tensor(param_array)
            state_dict[key].copy_(param_tensor)
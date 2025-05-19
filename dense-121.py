# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:58:59 2025

@author: szk9
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

## Hyperparameters
BATCH = 32
EPOCH = 60
classes = 10
## Growth rate defines the number of feature maps each layer in a dense block produces
growth_rate=32

## Loading dataset
data = h5py.File("C:/Combined/Work/My_dataset/UPDATED_CODES/dataset/data.h5", 'r')
x = np.asarray(data['dataset_gray'])
y = np.asarray(data['dataset_label'])

## The train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## Converting the data to torch tensors
x_train = torch.tensor(x_train[:, :, :, 0], dtype=torch.float32).unsqueeze(1)  # Remove last dimension and add channel dimension for grayscale
x_test = torch.tensor(x_test[:, :, :, 0], dtype=torch.float32).unsqueeze(1)  # Remove last dimension and add channel dimension for grayscale

y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

## Creating DataLoader for PyTorch
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

## The bottlneck layers
class BottleneckLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return torch.cat([x, out], 1)  ## Concatenate the input and the output
    
## Transition layers are used to connect dense blocks to reducing the number of feature maps and downsampling the spatial dimensions of the feature maps.
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out

## The denseblocks are the building blocks of DenseNet architectures
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(n_layers):
            layers.append(BottleneckLayer(in_channels + i * growth_rate, growth_rate))
        self.denseblock = nn.Sequential(*layers)

    def forward(self, x):
        return self.denseblock(x)

## Assembling the densenet model
class DenseNet(nn.Module):
    def __init__(self, block_config=(6, 12, 24, 16), growth_rate=32, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.conv1 = nn.Conv2d(1, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        num_channels = 2 * growth_rate
        self.dense_layers = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_channels, growth_rate, num_layers)
            self.dense_layers.append(block)
            num_channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                transition = TransitionLayer(num_channels, num_channels // 2)
                self.dense_layers.append(transition)
                num_channels = num_channels // 2

        self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        for layer in self.dense_layers:
            x = layer(x)

        x = F.relu(self.bn2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

## Example of how to create a DenseNet model (among DenseNet-121, DenseNet-169, DenseNet-201, and DenseNet-264, we will use densenet121)
def densenet121(num_classes=10):
    return DenseNet(block_config=(6, 12, 24, 16), growth_rate=32, num_classes=num_classes)

## Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet(block_config=(6, 12, 24, 16), num_classes=10).to(device)

## Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

## Displaying the  model parameters
num_params = count_parameters(model)
print(f"Total number of parameters: {num_params}")

## Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

## The training loop starts here..........................................................................
for epoch in range(EPOCH):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCH}], Loss: {running_loss/len(train_loader):.4f}")
    scheduler.step(running_loss/len(train_loader))


## Evaluating the model on test data
    model.eval()
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f"Test Accuracy after Epoch {epoch+1}: {test_accuracy:.2f}%")

## confusion matrix visualization
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[f"Class {i}" for i in range(classes)], yticklabels=[f"Class {i}" for i in range(classes)])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

## classification report visualization
class_report = classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(classes)])
print("Classification Report:")
print(class_report)

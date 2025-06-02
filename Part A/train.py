import numpy
import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Define the CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, num_filters=32, filter_multiplier=1, dropout=0.2, batch_norm=False, dense_size=64, num_classes=10,activation='relu'):
        super(CNNClassifier, self).__init__()

        
        self.activation = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'mish': nn.Mish()
        }[activation.lower()]


        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB Images

        for i in range(5):
            out_channels = num_filters
            conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.conv_layers.append(conv_layer)

            if batch_norm:
                self.conv_layers.append(nn.BatchNorm2d(out_channels))

            self.conv_layers.append(self.activation)
           
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = out_channels
            num_filters = int(num_filters * filter_multiplier)


        self.fc1 = nn.Linear(in_channels * 8 * 8, dense_size)

        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_size, num_classes)


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = torch.flatten(x, start_dim=1)

    # Get the dynamic input size for fc1
        if not hasattr(self, 'fc1'):
            self.fc1 = nn.Linear(x.shape[1], self.dense_size).to(x.device)

        x = self.fc1(x)
        x = self.dropout(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

# Dataset Preparation
def prepare_dataset(data_dir="inaturalist_12K", augment_data=False, batch_size=256):
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "val")

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all images to 256x256
        transforms.RandomRotation(30) if augment_data else transforms.Lambda(lambda x: x),
        transforms.RandomHorizontalFlip() if augment_data else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize all images to 256x256
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])


    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Train function
def train():
    config_defaults = {
        "num_filters": 32,
        "filter_multiplier": 2,
        "augment_data": False,
        "dropout": 0.3,
        "batch_norm": False,
        "epochs": 10,
        "dense_size": 64,
        "lr": 0.001,
        "activation": "relu"
    }

    wandb.init(project="CNN-Hyperparam-Tuning", config=config_defaults)
    config = wandb.config
    wandb.run.name = set_run_name(config.num_filters, config.filter_multiplier, config.augment_data, config.dropout, config.batch_norm)

    # Prepare dataset
    train_loader, val_loader, _ = prepare_dataset(augment_data=config.augment_data)

    # Define model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(
        num_filters=config.num_filters,
        filter_multiplier=config.filter_multiplier,
        dropout=config.dropout,
        batch_norm=config.batch_norm,
        dense_size=config.dense_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Training Loop
    for epoch in range(config.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100. * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        # Log to WandB
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    print("Training Completed!")

# Function to create readable run names
def set_run_name(num_filters=32, filter_multiplier=1, augment_data=False, dropout=0.2, batch_norm=False):
    augment_data_options = {True: "Y", False: "N"}
    batch_norm_options = {True: "Y", False: "N"}
    return f"num_{num_filters}_org_{filter_multiplier}_aug_{augment_data_options[augment_data]}_drop_{dropout}_norm_{batch_norm_options[batch_norm]}"


# Sweep Configuration
Sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric": {
    "name": "val_accuracy",
    "goal": "maximize"
  },
  "parameters": {
    "activation": {"values": ["Relu", "Gelu", "silu", "Mish"]},
    "filter_multiplier": {"values": [(2,2), (3,3), (4,4)]},
    "batch_size": {"values": [32, 64]},
    "padding": {"values": ["same", "valid"]},
    "augment_data": {"values": [True, False]},
    "optimizer": {"values": ["sgd", "adam", "rmsprop", "nadam"]},
    "batch_norma": {"values": [True, False]},
    "num_filters": {"values": [32, 64]},
    "dense_size": {"values": [32, 64, 128]},
    "dropout": {"values": [None, 0.2, 0.3]},
    "epochs": {"values": [10]}
  }
}


# Run training
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project='DA6401-Assignment2', entity='ma23c044-indian-institute-of-technology-madras')
    wandb.agent(sweep_id, function=train, count=10) 



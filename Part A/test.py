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
            # self.conv_layers.append(nn.ReLU())
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



# Function to prepare the test dataset
def prepare_test_dataset(DATA_DIR="inaturalist_12K", augment_data=False, image_size=256, batch_size=30):
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "val")

    transform_train = transforms.Compose([
        transforms.RandomRotation(90) if augment_data else transforms.Lambda(lambda x: x),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip() if augment_data else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



def log_test_results_to_wandb(images, predictions, labels):
    output_map = {0: 'Amphibia', 1: 'Animalia', 2: 'Arachnida', 3: 'Aves', 4: 'Fungi',
                  5: 'Insecta', 6: 'Mammalia', 7: 'Mollusca', 8: 'Plantae', 9: 'Reptilia'}

    table = wandb.Table(columns=["Image", "Predicted", "Actual"])

    for i in range(len(images)):
        img = images[i]
        img = img * 0.5 + 0.5  # unnormalize
        np_img = img.permute(1, 2, 0).numpy()
        pred_label = output_map[predictions[i]]
        true_label = output_map[labels[i]]
        table.add_data(wandb.Image(np_img), pred_label, true_label)

    wandb.log({"Test Predictions": table})




# Plot test results
def plot_test_results(images, predictions, labels):
    fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(15, 15))
    output_map = {0: 'Amphibia', 1: 'Animalia', 2: 'Arachnida', 3: 'Aves', 4: 'Fungi',
                  5: 'Insecta', 6: 'Mammalia', 7: 'Mollusca', 8: 'Plantae', 9: 'Reptilia'}

    for i in range(30):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        ax[i // 6, i % 6].imshow(img)
        ax[i // 6, i % 6].axis('off')
        ax[i // 6, i % 6].set_title(f"Predicted: {output_map[predictions[i].item()]}\nLabel: {output_map[labels[i].item()]}")
    plt.show()

# Visualize feature maps from the first convolutional layer
def plot_filters(model, image):
    sub_model = nn.Sequential(*list(model.children())[:2])  # Extract first conv layer
    feature_maps = sub_model(image.unsqueeze(0)).detach().cpu().squeeze()
    fig, ax = plt.subplots(4, 8, figsize=(12, 6))
    for i in range(feature_maps.shape[0]):
        ax[i // 8, i % 8].imshow(feature_maps[i], cmap='gray')
        ax[i // 8, i % 8].axis('off')
    plt.show()

# Function to set run name
def set_run_name(num_filters, filter_multiplier, augment_data, dropout, batch_norm):
    return f"num_{num_filters}_org_{filter_multiplier}_aug_{augment_data}_drop_{dropout}_norm_{batch_norm}"

# Define test function
def test():
    config_defaults = {
        "num_filters": 32,
        "filter_multiplier": 2,
        "augment_data": False,
        "dropout": 0.3,
        "batch_norm": False,
        "epochs": 3,
        "dense_size": 64,
        "activation": "relu",
        "lr": 0.001
    }

    wandb.init(config=config_defaults, project="CNN-Test")
    config = wandb.config
    wandb.run.name = set_run_name(config.num_filters, config.filter_multiplier, config.augment_data, config.dropout, config.batch_norm)

    train_loader,test_loader = prepare_test_dataset(augment_data=config.augment_data, image_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(
                    num_filters=config.num_filters,
                    filter_multiplier=config.filter_multiplier,
                    dropout=config.dropout,
                    batch_norm=config.batch_norm,
                    dense_size=config.dense_size,
                    num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # Train the model
    model.train()
    for epoch in range(config.epochs):
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
        train_acc = 100. * correct / total
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})
        print(f"Epoch {epoch+1}/{config.epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
   
    # Evaluate
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    all_images = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)


            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_images.extend(images.cpu())


    accuracy = 100. * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    wandb.log({"test_accuracy": accuracy})

    # Visualize first 30 predictions
    plot_test_results(all_images[:30], all_preds[:30], all_labels[:30])

    # log_filters_to_wandb(all_images[:30], all_preds[:30], all_labels[:30])
    log_test_results_to_wandb(all_images[:30], all_preds[:30], all_labels[:30])

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'num_filters': {
            'values': [16, 32]
        },
        'filter_multiplier': {
            'values': [1, 2]
        },
        'augment_data': {
            'values': [False, True]
        },
        'dropout': {
            'values': [0.3, 0.5]
        },
        'batch_norm': {
            'values': [False, True]
        },
        'epochs': {
            'value': 3
        },
        'dense_size': {
            'values': [64, 128]
        },
        'activation': {
            'values': ['relu']
        },
        'lr': {
            'values': [0.001, 0.0001]
        }
    }
}
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="CNN-Test")
    wandb.agent(sweep_id, function=test, count=10)  



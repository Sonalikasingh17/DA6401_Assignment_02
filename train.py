import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the CNN model class
class CNNClassifier(nn.Module):
    def __init__(self, num_filters=[32, 64, 128, 256, 512], kernel_sizes=[3, 3, 3, 3, 3], activation=F.relu, dense_neurons=128, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.activation = activation
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer with 2x2 kernel

        in_channels = 3  # Assuming RGB images (3 channels)
        for i in range(5):  # 5 convolutional layers
            self.conv_layers.append(nn.Conv2d(in_channels, num_filters[i], kernel_size=kernel_sizes[i], padding=1))
            in_channels = num_filters[i]  # Update input channels for next layer

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[-1] * 4 * 4, dense_neurons)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(dense_neurons, num_classes)  # Output layer

    def forward(self, x):
        for conv in self.conv_layers:
            x = self.pool(self.activation(conv(x)))  # Conv -> Activation -> MaxPool
        x = torch.flatten(x, 1)  # Flatten before fully connected layer
        x = self.activation(self.fc1(x))  # Dense layer with activation
        x = self.fc2(x)  # Output layer
        return x

# Example usage
model = CNNClassifier()
print(model)


# Hyperparameter tuning configuration
# This is a placeholder configuration for hyperparameter tuning.

sweep_config = {
    'method': 'random',  # Other options: 'grid', 'bayesian'
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'num_filters': {'values': [32, 64, 128]},
        'activation': {'values': ['ReLU', 'GELU', 'SiLU', 'Mish']},
        'filter_organization': {'values': ["same", "double", "half"]},
        'data_augmentation': {'values': ["Yes", "No"]},
        'batch_norm': {'values': ["Yes", "No"]},
        'dropout_rate': {'values': [0.2, 0.3]}
    }
}

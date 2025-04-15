import numpy
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision.models import (
    resnet50, googlenet, inception_v3, vgg16, efficientnet_v2_s, vit_b_16)
from torchmetrics import Accuracy
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import os
import wandb
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import ObjectDetectionModel  

IMG_SIZE = (128, 128)

def get_dataloaders(data_dir, batch_size, img_size, data_augmentation):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ]) if data_augmentation else transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])

    full_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=transform_train)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_test  # Use test transform for val

    test_dataset = ImageFolder(os.path.join(data_dir, "val"), transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader
'''
sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric": {
    "name": "val_accuracy",
    "goal": "maximize"
  },
  "early_terminate": {
    "type": "hyperband",
    "min_iter": 3,
    "s": 2
  },
  "parameters": {
    "activation": {"values": ["relu", "elu", "selu"]},
    "filter_size": {"values": [(2,2), (3,3), (4,4)]},
    "batch_size": {"values": [32, 64]},
    "padding": {"values": ["same", "valid"]},
    "data_augmentation": {"values": [True, False]},
    "optimizer": {"values": ["sgd", "adam", "rmsprop", "nadam"]},
    "batch_normalization": {"values": [True, False]},
    "batch_normalisation_location": {"values": ["Before", "After"]},
    "number_of_filters_base": {"values": [32, 64]},
    "dense_neurons": {"values": [32, 64, 128]},
    "dropout_location": {"values": ["conv", "dense", "all"]},
    "dropout_fraction": {"values": [None, 0.2, 0.3]},
    "global_average_pooling": {"values": [False, True]},
  }
}

sweep_id = wandb.sweep(sweep_config, project='DA6401-Assignment2', entity='ma23c044-indian-institute-of-technology-madras')
wandb.agent(sweep_id, function=train)
'''
def train():
    # Sweep or default config
    config_defaults = dict(
        num_hidden_cnn_layers=5,
        activation='relu',
        batch_normalization=True,
        batch_normalisation_location="After",
        filter_distribution="double",
        filter_size=(3, 3),
        number_of_filters_base=32,
        dropout_fraction=None,
        dropout_location="dense",
        pool_size=(2, 2),
        padding='same',
        dense_neurons=128,
        num_classes=10,
        optimizer='adam',
        epochs=5,
        batch_size=32,
        data_augmentation=False,
        global_average_pooling=True,
        img_size=IMG_SIZE,
        base_model="RN50",
        using_pretrained_model=True
    )

    # Initialize wandb
    wandb.init(project='DA6401-Assignment2', config=config_defaults, entity='ma23c044-indian-institute-of-technology-madras')
    config = wandb.config

    # Custom run name just like in your TF code
    wandb.run.name = f"OBJDET_{config.num_hidden_cnn_layers}_dn_{config.dense_neurons}_opt_{config.optimizer}_dro_{config.dropout_fraction}_bs_{config.batch_size}_fd_{config.filter_distribution}_bnl_{config.batch_normalisation_location}_dpl_{config.dropout_location}"

    wandb_logger = WandbLogger(log_model="all")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders('./inaturalist_12K', config.batch_size, config.img_size, config.data_augmentation)

    # Model
    model = ObjectDetectionModel(
        IMG_SIZE=config.img_size,
        modelConfigDict=dict(config),
        using_pretrained_model=config.using_pretrained_model,
        base_model=config.base_model
    )

    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator='auto',
        logger=wandb_logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")],
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader, val_loader)

    # Save model checkpoint
    os.makedirs("./TrainedModel", exist_ok=True)
    trainer.save_checkpoint(f"./TrainedModel/{wandb.run.name}.ckpt")

    wandb.finish()

if __name__ == "__main__":
    train()


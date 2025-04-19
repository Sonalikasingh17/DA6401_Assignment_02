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
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt


BASE_MODELS = {
                      "RN50": resnet50,
                      "IV3": inception_v3,
                      "GOOGLENET": googlenet,
                      "VGG16": vgg16,
                      "EFFICIENTNETV2": efficientnet_v2_s,
                      "VIT": vit_b_16
                  }

class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, IMG_SIZE, modelConfigDict, using_pretrained_model=False, base_model="RN50"):
        super().__init__()

        self.save_hyperparameters()
      
        self.modelConfigDict = modelConfigDict
        self.IMG_HEIGHT = IMG_SIZE[0]
        self.IMG_WIDTH = IMG_SIZE[1]
        self.input_channels = 3
        self.input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)


        self.num_hidden_cnn_layers= modelConfigDict["num_hidden_cnn_layers"]
        self.activation_fn = self._get_activation_fn(modelConfigDict["activation"])
        self.batch_normalization = modelConfigDict["batch_normalization"]
        self.filter_distribution = modelConfigDict["filter_distribution"]
        self.filter_size = modelConfigDict["filter_size"]
        self.number_of_filters_base  = modelConfigDict["number_of_filters_base"]
        self.dropout_fraction = modelConfigDict["dropout_fraction"]
        self.pool_size = modelConfigDict["pool_size"]
        self.padding = modelConfigDict["padding"]
        self.dense_neurons = modelConfigDict["dense_neurons"]
        self.num_classes = modelConfigDict["num_classes"]
        self.optimizer = modelConfigDict["optimizer"]
        self.batch_normalisation_location = modelConfigDict["batch_normalisation_location"]


        self.using_pretrained_model = using_pretrained_model
        self.model = self._build_with_pretrained_model(base_model)
        
        self.accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

    def _get_activation_fn(self, name):
        return {
            "ReLU": nn.ReLU(),
            "GELU": nn.GELU(),
            "SiLU": nn.SiLU(),
            "Mish": nn.Mish()
        }.get(name.lower(), nn.ReLU())

    def _build_with_pretrained_model(self, model_name):
        base_model = BASE_MODELS[model_name](pretrained=True)
        for param in base_model.parameters():
            param.requires_grad = False


        dense_neurons = self.modelConfigDict["dense_neurons"]

        if model_name == "RN50":
            num_feats = base_model.fc.in_features
            base_model.fc = nn.Sequential(
                nn.Linear(num_feats, dense_neurons),
                nn.ReLU(),
                nn.Linear(dense_neurons, self.num_classes)
            )

        elif model_name == "VGG16":
            num_feats = base_model.classifier[6].in_features
            base_model.classifier[6] = nn.Sequential(
                nn.Linear(num_feats, dense_neurons),
                nn.ReLU(),
                nn.Linear(dense_neurons, self.num_classes)
            )

        elif model_name in ["IV3", "GOOGLENET"]:
            num_feats = base_model.fc.in_features
            base_model.fc = nn.Sequential(
                nn.Linear(num_feats, dense_neurons),
                nn.ReLU(),
                nn.Linear(dense_neurons, self.num_classes)
            )
            if model_name == "IV3":
                base_model.aux_logits = False  # turn off auxiliary logits if not used

        elif model_name == "EFFICIENTNETV2":
            num_feats = base_model.classifier[1].in_features
            base_model.classifier[1] = nn.Sequential(
                nn.Linear(num_feats, dense_neurons),
                nn.ReLU(),
                nn.Linear(dense_neurons, self.num_classes)
            )

        elif model_name == "VIT":
            num_feats = base_model.heads.head.in_features
            base_model.heads.head = nn.Sequential(
                nn.Linear(num_feats, dense_neurons),
                nn.ReLU(),
                nn.Linear(dense_neurons, self.num_classes)
            )

        else:
            raise ValueError(f"Base model {model_name} not supported.")

        return base_model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits.softmax(dim=1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.accuracy(logits.softmax(dim=1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        opt_name = self.modelConfigDict["optimizer"]
        lr = self.modelConfigDict.get("learning_rate", 1e-3)
        if opt_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif opt_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            return torch.optim.Adam(self.parameters(), lr=lr)


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

def train():
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
        img_size=IMG_SIZE,
        base_model="RN50",
        using_pretrained_model=True
    )

    # Initialize wandb
    wandb.init(project='DA6401-Assignment2', config=config_defaults, entity='ma23c044-indian-institute-of-technology-madras')
    config = wandb.config

    # Custom run name just like in your TF code
    wandb.run.name = f"hl_CNN_{config.num_hidden_cnn_layers}_dn_{config.dense_neurons}_opt_{config.optimizer}_dro_{config.dropout_fraction}_bs_{config.batch_size}_fd_{config.filter_distribution}_bnl_{config.batch_normalisation_location}_dpl_{config.dropout_location}"

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

    wandb.finish()
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
    "activation": {"values": ["Relu", "Gelu", "Silu", "Mish]},
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
  }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment2", entity='ma23c044-indian-institute-of-technology-madras') 
    wandb.agent(sweep_id, function=train, count=10)  



import numpy
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
import torchvision
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
from pytorch_lightning.loggers import WandbLogger



BASE_MODELS = {
    "RN50": resnet50,
    "IV3": inception_v3,
    "GOOGLENET": googlenet,
    "VGG16": vgg16,
    "EFFICIENTNETV2": efficientnet_v2_s,
}

DATA_DIR = "./inaturalist_12K"
IMG_SIZE = (224, 224)
NUM_CLASSES = 10


# -------------------  Transforms & Dataloaders -------------------
def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
        ])

def get_dataloaders(batch_size, augment):
    transform = get_transforms(augment)
    full_dataset = ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    test_dataset = ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


# -------------------  TransferModel with Fine-Tuning Strategy -------------------
class TransferModel(pl.LightningModule):
    def __init__(self, base_model_name='RN50', dense_neurons=256, optimizer_name='adam', lr=1e-3,
                 finetune_strategy='freeze_all', unfreeze_k=0):
        super().__init__()
        self.save_hyperparameters()

        base_model = BASE_MODELS[base_model_name](weights='IMAGENET1K_V1')
        self.finetune_strategy = finetune_strategy

    
        

        # Handle model structure for final feature layer
        if hasattr(base_model, 'fc') and isinstance(base_model.fc, nn.Linear):
            # ResNet, GoogLeNet, InceptionV3
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()

        elif hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Sequential):
            # VGG, AlexNet
            in_features = base_model.classifier[-1].in_features
            base_model.classifier = nn.Identity()

        elif hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Module):
            # EfficientNetV2 or MobileNet
            linear_layers = [m for m in base_model.classifier.modules() if isinstance(m, nn.Linear)]
            if len(linear_layers) == 1:
                in_features = linear_layers[0].in_features
                base_model.classifier = nn.Identity()
            else:
                raise ValueError(f"Couldn't find a single Linear layer in {base_model}. Please check the model definition.")

        else:
            raise ValueError(f"Unknown model structure for: {base_model}")


        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(in_features, dense_neurons),
            nn.ReLU(),
            nn.Linear(dense_neurons, NUM_CLASSES)
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.configure_finetune(finetune_strategy, unfreeze_k)

    def configure_finetune(self, strategy, k):
        all_params = list(self.base_model.parameters())
        if strategy == 'freeze_all':
            for p in all_params:
                p.requires_grad = False
        elif strategy == 'unfreeze_all':
            for p in all_params:
                p.requires_grad = True
        elif strategy == 'unfreeze_last_k':
            for p in all_params[:-k]:
                p.requires_grad = False
            for p in all_params[-k:]:
                p.requires_grad = True
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

    def forward(self, x):
        if self.base_model == "inception_v3":
           outputs = self.model(x)
           if isinstance(outputs, tuple):
              return outputs[0]  # return only the main output
           return outputs
        else:
            return self.base_model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        lr = self.hparams.lr
        if self.hparams.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif self.hparams.optimizer_name == 'nadam':
            return torch.optim.NAdam(self.parameters(), lr=lr)
        elif self.hparams.optimizer_name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

# -------------------  Training with WandB Sweep -------------------
def train_wandb():
    wandb.init(project="iNat12k-transfer", job_type="sweep", entity = "ma23c044-indian-institute-of-technology-madras")
    config = wandb.config
              
    wandb.run.name=f"bm_{wandb.config.base_model}_opt_{wandb.config.optimizer}_lr_{wandb.config.lr:.1e}_strat_{wandb.config.finetune_strategy}_dn_{wandb.config.dense_neurons}_bs_{wandb.config.batch_size}_aug_{wandb.config.augment}"



    # Log sweep strategy
    wandb.log({
        'finetune_strategy': config.finetune_strategy,
        'unfreeze_k': config.unfreeze_k if config.finetune_strategy == 'unfreeze_last_k' else 0
    })

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.batch_size,
        augment=config.augment
    )

    model = TransferModel(
        base_model_name=config.base_model,
        dense_neurons=config.dense_neurons,
        optimizer_name=config.optimizer,
        lr=config.lr,
        finetune_strategy=config.finetune_strategy,
        unfreeze_k=config.unfreeze_k
    )

    wandb_logger = WandbLogger()
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=[EarlyStopping(monitor='val_acc', mode='max', patience=3)],
        accelerator='auto'
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    wandb.finish()


# -------------------  Sweep Config -------------------
sweep_config = {
    'method': 'random',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'base_model': {'values': ['RN50', 'VGG16', 'GOOGLENET','EFFICIENTNETV2','IV3']},
        'dense_neurons': {'values': [128, 256]},
        'optimizer': {'values': ['adam', 'nadam', 'rmsprop']},
        'lr': {'min': 1e-5, 'max': 1e-3},
        'batch_size': {'values': [32, 64]},
        'augment': {'values': [True, False]},
        'epochs': {'value': 3},
        'finetune_strategy': {'values': ['freeze_all', 'unfreeze_all', 'unfreeze_last_k']},
        'unfreeze_k': {'values': [5, 10, 20]}
    }
}

# Launch sweep
sweep_id = wandb.sweep(sweep_config, project="iNat12k-transfer",entity = "ma23c044-indian-institute-of-technology-madras")
wandb.agent(sweep_id, function=train_wandb, count=5)

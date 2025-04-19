# DA6401_Assignment_02
** The link to the wandb report: **

https://wandb.ai/ma23c044-indian-institute-of-technology-madras/inat-cnn-sweep/reports/DA6401-Assignment-02--VmlldzoxMjM0MjI5OQ
---
## PART A:  CNN Hyperparameter Tuning on iNaturalist12K with PyTorch from Scratch
This project implements a **5-layer configurable CNN classifier** using **PyTorch** and performs **hyperparameter optimization** via **Weights & Biases (WandB) Sweeps** on the iNaturalist 12K dataset.

---
### Training Strategy Used
Trained a **CNN model from scratch** with fully configurable architecture:

-  **Number of Conv Layers**: 5
-  **Filter Multiplier**: Scales filters per layer (e.g., 32 → 64 → 128 …)
-  **Activation Functions**: ReLU, GELU, SiLU, Mish
-  **Batch Normalization**: Optional
-  **Dropout Regularization**: Optional
-  **Dense Layer Size**: Configurable

---
### Training Configuration

| Parameter              | Value(s) Used                                      |
|------------------------|----------------------------------------------------|
| **Framework**          | PyTorch                                            |
| **Dataset**            | iNaturalist 12K                                    |
| **Input Size**         | 256x256                                            |
| **Augmentation**       | Random Rotation + Horizontal Flip (configurable)  |
| **Loss Function**      | CrossEntropyLoss                                   |
| **Optimizers**         | SGD, Adam, RMSProp, Nadam (Swept)                  |
| **Learning Rate**      | Configurable                                       |
| **Epochs**             | 10                                                 |
| **Batch Size**         | 32, 64                                             |
| **Tracking**           | [Weights & Biases (WandB)](https://wandb.ai)      |
---

### Transform Data:
- Resized to `(256, 256)`
- Normalized using mean `0.5` and std `0.5`
- Augmentation enabled via flags

---
### Training Script Highlights (`train.py`)
 **Modular CNN class** with:
- 5 configurable convolutional layers  
- Optional Batch Normalization  
- Flexible activation functions  
- Dropout + Dense classification head  

 **Train/Validation split** with 90/10 ratio  
**Logs metrics** (loss/accuracy) to WandB  

##  WandB Sweep Configuration
We used **Bayesian Optimization** to tune the following:

```yaml
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  activation: [ReLU, GELU, SiLU, Mish]
  filter_multiplier: [(2,2), (3,3), (4,4)]
  batch_size: [32, 64]
  augment_data: [True, False]
  optimizer: [sgd, adam, rmsprop, nadam]
  batch_norma: [True, False]
  num_filters: [32, 64]
  dense_size: [32, 64, 128]
  dropout: [None, 0.2, 0.3]
  epochs: [10]
```

### Example Run Name (Auto-generated):
```
num_32_org_2_aug_Y_drop_0.3_norm_Y
```
---

##  Testing Script (`test.py`)
- Loads the trained model  
- Evaluates on validation/test data  
- Computes final accuracy and loss  
- Visualizes predictions using 'matplotlib' and 'wandb'

---

### Requirement
```bash
pip install torch torchvision wandb matplotlib
```
---
## PART B : Transfer Learning on iNaturalist 12K with PyTorch Lightning
This repository implements a transfer learning pipeline using various pretrained models on the iNaturalist 12K dataset. It supports model configuration, data augmentation, fine-tuning strategies, and hyperparameter sweeps using Weights & Biases (WandB).

### Key Features
- **Transfer learning with pretrained models:**
  - `ResNet50`
  - `InceptionV3`
  - `GoogLeNet`
  - `VGG16`
  - `EfficientNetV2-S`
  - 
- **Configurable fine-tuning strategies:**
  - `freeze_all`: Freeze all base model layers.
  - `unfreeze_all`: Train all layers.
  - `unfreeze_last_k`: Unfreeze last k layers.

- Supports **data augmentation**
- **Hyperparameter sweeps** via **Weights & Biases (WandB)** for optimal configuration
-  Built using **PyTorch Lightning** for clean, modular training workflows
---
### Project Structure

### `get_transforms()`
- Defines image preprocessing and optional data augmentation.

###  `get_dataloaders()`
- Prepares train, validation, and test `DataLoader`s using the iNaturalist dataset.

###  `TransferModel`
- PyTorch Lightning model wrapping a base pretrained network with:
  - A custom classifier head.
  - Configurable fine-tuning strategies (`freeze_all`, `unfreeze_all`, `unfreeze_last_k`).
  - Support for multiple optimizers: `adam`, `nadam`, `rmsprop`.

###  Training, Validation, and Testing
- Implements standard PyTorch Lightning methods:
  - `training_step`
  - `validation_step`
  - `test_step`
---

### Dataset:
Image Size: Resized to (224, 224)
Classes: 10
Transforms:
Normal: Resize + ToTensor
Augmented: + Horizontal Flip, Random Rotation (15°)
---
## WandB Sweep Configuration
We use **Weights & Biases (WandB)** to perform a random search over hyperparameters like:
- Base model
- Optimizer
- Learning rate
- Batch size
- Data augmentation toggle
- Fine-tune strategy and number of layers to unfreeze

Sweep Configuration
```yaml
parameters:
  base_model: ['RN50', 'VGG16', 'GOOGLENET', 'EFFICIENTNETV2', 'IV3']
  optimizer: ['adam', 'nadam', 'rmsprop']
  lr: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
  batch_size: [32, 64]
  augment: [True, False]
  dense_neurons: [128, 256]
  finetune_strategy: ['freeze_all', 'unfreeze_all', 'unfreeze_last_k']
  unfreeze_k: [5, 10, 20]
```
### Best Model Configuration (from sweep)
```yaml
Base Model       : EFFICIENTNETV2  
Optimizer        : Nadam  
Learning Rate    : 7.7e-04  
Fine-Tune Strat  : freeze_all  
Dense Neurons    : 128  
Batch Size       : 64  
Data Augment     : True  
```
### Wandb run name
```
bm_EFFICIENTNETV2_opt_nadam_lr_7.7e-04_strat_freeze_all_dn_128_bs_64_aug_True
```
### Requirements
```bash
pip install torch torchvision pytorch-lightning wandb matplotlib
```


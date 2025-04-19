# DA6401_Assignment_02
## PaRT A
### CNN Hyperparameter Tuning on iNaturalist12K with PyTorch from Scratch
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




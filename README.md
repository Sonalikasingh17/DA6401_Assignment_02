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

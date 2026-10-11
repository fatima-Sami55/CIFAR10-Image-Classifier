# Deep Learning for Perception (Fall 2026)
## Assignment 2 (Section B): Robust CIFAR-10 Image Classification with Limited Data

**Student:** Fatima Sami  
**Roll Number:** 23F-0791  
**Batch:** 23F  
**Repository URL:** [https://github.com/fatima-Sami55/CIFAR10-Image-Classifier.git](https://github.com/fatima-Sami55/CIFAR10-Image-Classifier.git)

---

## Project Overview

This repository contains the complete implementation for **Section B of Assignment 2 (Deep Learning for Perception)**. The core objective is to design, train, and evaluate a robust Convolutional Neural Network (CNN) trained from scratch on only **500 images (50 per class)** from CIFAR-10 and evaluate its generalization ability on a severely corrupted **Shifted Test Set** (10,000 images).

### Challenges
1. **Extreme Data Scarcity:** Training with only 400 clean images (40 per class) and 100 validation images (10 per class) leads to extreme variance and rapid overfitting.
2. **Severe Covariate Shift:** The 10,000 test images are subjected to a fixed, seeded corruption pipeline comprising:
   - Random rotations up to $\pm 25^\circ$
   - Photometric brightness and contrast shifts ($0.5\times$ to $1.5\times$)
   - Random rectangular occlusions ($\approx 25\%$ image area)
   - An undisclosed hidden corruption

---

## Architectural Design

To adhere to the assignment constraints while maximizing multi-scale feature representation:
- **Parameter Budget:** Hard constraint of $\le 1,000,000$ trainable parameters. Our architecture contains **87,778 trainable parameters** (~8.8% of the budget), preventing excessive parameter bloat on tiny data.
- **Inception-Style Multi-Branch Modules:**
  - Parallel $1\times 1$, $3\times 3$, and $5\times 5$ convolutions with $1\times 1$ bottleneck channel reductions.
  - Parallel $3\times 3$ max-pooling branch with $1\times 1$ projection.
- **Stem & Classification Head:**
  - LeNet/AlexNet-style convolutional stem ($3 \to 32 \to 64$ channels).
  - Global Average Pooling (GAP) replacing parameter-heavy fully connected layers.
  - Head consists of a single hidden dense layer (128 units), Dropout ($p=0.5$), and a 10-way linear classifier.
- **Training Recipe:**
  - Optimization: SGD with Nesterov momentum ($0.9$).
  - Learning Rate: Initial LR of $0.05$ with Cosine Annealing schedule over 60 epochs.
  - Weight Decay: $5 \times 10^{-4}$ applied exclusively to weights (excluding BatchNorm and biases).
  - Regularization: Batch Normalization after every conv layer, Dropout ($0.5$), and Early Stopping (patience 15).

---

## Key Experimental Results

All experiments are conducted across **3 fixed seeds (0, 1, 2)** and reported as **Mean $\pm$ Std**.

### Mandatory Ablation Study (Part D)

| Configuration | Validation Accuracy (Clean) | Clean Test Accuracy | Shifted Test Accuracy | Shifted Gain vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **No augmentation (Baseline)** | $44.7\% \pm 0.6\%$ | $40.8\% \pm 2.1\%$ | $16.5\% \pm 1.6\%$ | Baseline ($+0.0\%$) |
| **Geometric only** (crop, flip, rotation) | $33.3\% \pm 1.2\%$ | $35.3\% \pm 1.4\%$ | $21.9\% \pm 1.4\%$ | **$+5.4\%$** |
| **Photometric only** (color jitter, blur, noise) | $35.3\% \pm 7.5\%$ | $35.1\% \pm 5.5\%$ | $18.8\% \pm 0.7\%$ | $+2.3\%$ |
| **Geometric + photometric** | $28.7\% \pm 2.1\%$ | $28.0\% \pm 0.5\%$ | $20.1\% \pm 2.3\%$ | $+3.6\%$ |
| **+ Cutout / Random Erasing** | $27.3\% \pm 5.1\%$ | $26.7\% \pm 6.0\%$ | $20.8\% \pm 5.0\%$ | $+4.3\%$ |
| **+ CutMix + label smoothing (Top Model)** | $30.3\% \pm 2.5\%$ | $30.3\% \pm 1.7\%$ | **$25.1\% \pm 1.0\%$** | **$+8.6\%$** |

### Summary of Findings
1. **Winning Configuration:** Composite pipeline combining Geometric transforms (rotation $\pm 25^\circ$, translation $\pm 4$ px, scale $0.9\text{--}1.1$, horizontal flip) + Photometric transforms (brightness, contrast, saturation, blur, noise) + Cutout + CutMix with Label Smoothing ($0.1$). It achieved **$25.1\% \pm 1.0\%$** on the shifted test set.
2. **Single Change Contributing Most:** **Geometric Augmentation** delivered the largest single performance leap ($+5.4$ percentage points, lifting accuracy from $16.5\%$ to $21.9\%$). This directly overcomes the rotational covariate shift in the shifted test set.
3. **CutMix + Label Smoothing Impact:** Adding CutMix and label smoothing provided the second largest single jump ($+4.3$ points from $20.8\%$ to $25.1\%$), mitigating rectangular occlusion and smoothing decision boundaries.
4. **Baseline Failure Diagnosis (Part A):**
   - *Overfitting Gap:* Train accuracy was $88.9\% \pm 11.3\%$ vs. clean test $40.8\% \pm 2.1\%$ (a $48.1\%$ gap).
   - *Covariate Shift Gap:* Clean test accuracy was $40.8\% \pm 2.1\%$ vs. shifted test $16.5\% \pm 1.6\%$ (a $24.3\%$ drop).
   - The unregularized baseline failed due to both severe overfitting and severe covariate shift.
5. **Test-Time Augmentation (Bonus Part F):** Averaging softmax predictions over 8 transformed views (original, flip, 4 directional translations, 2 flipped shifts) provided consistent predictions at an $8.2\times$ inference runtime factor.

---

## Repository Structure

```
.
|-- DL_ASS02_23F_0791.ipynb                # Fully executed Jupyter Notebook with all outputs & plots
|-- F230791 Fatima - Assignment#2 DLP Summary.docx # One-page executive Word summary report
|-- results.json                           # Raw metric arrays across all seeds and ablation stages
|-- .gitignore                             # Excludes large datasets and cache files
`-- README.md                              # Project documentation and reproduction guide
```

---

## Instructions to Reproduce

### 1. Requirements & Dependencies
Ensure you have Python 3.9+ installed with the following packages:
```bash
pip install torch torchvision numpy matplotlib pandas
```
*(The codebase works seamlessly on both CPU and CUDA-enabled GPUs).*

### 2. Dataset Setup
The notebook uses `torchvision.datasets.CIFAR10(root="./data", download=True)`, which automatically downloads CIFAR-10 upon first execution. No manual downloading or pre-extracted data files are required.

### 3. Execution
1. Clone this repository:
   ```bash
   git clone https://github.com/fatima-Sami55/CIFAR10-Image-Classifier.git
   cd CIFAR10-Image-Classifier
   ```
2. Launch Jupyter Notebook / JupyterLab:
   ```bash
   jupyter notebook DL_ASS02_23F_0791.ipynb
   ```
3. Run all cells sequentially. The seed management ensures deterministic replication:
   - Data sampling: `DATA_SEED = 42`
   - Test shift pipeline: `SHIFT_SEED = 123`
   - Validation shift pipeline: `VAL_SHIFT_SEED = 7`
   - Model runs: `RUN_SEEDS = [0, 1, 2]`

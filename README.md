# CIFAR-10 Robust Image Classifier: Sample-Efficient Learning Under Covariate Shift

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sample-efficient, robust computer vision pipeline designed to tackle **extreme data scarcity (only 500 training images)** and **severe test-time covariate shift** (spatial rotations, photometric jitter, and random occlusions) on CIFAR-10. Built with an Inception-inspired multi-scale convolutional architecture trained strictly from scratch without any pretrained weights.

---

## Project Overview

In real-world computer vision deployments, collecting large labeled datasets is often expensive or impractical, and models deployed in the wild frequently encounter corrupted or out-of-distribution inputs. 

This project explores how architectural modularity, on-the-fly dynamic data augmentations, and regularized optimization strategies can enable a compact model (~87k parameters) trained on only **500 images (50 per class)** to generalize against a heavily perturbed **Shifted Test Set** (10,000 images).

### Key Challenges
1. **Extreme Sample Scarcity:** Training with only 400 clean images (40 per class) and 100 validation images (10 per class) leads to extreme variance and rapid memorization/overfitting.
2. **Severe Covariate Shift:** The 10,000 evaluation images undergo a rigorous multi-stage corruption pipeline:
   - Continuous random rotations up to $\pm 25^\circ$
   - Photometric brightness and contrast variations ($0.5\times$ to $1.5\times$)
   - Random rectangular occlusions ($\approx 25\%$ image area)
   - Hidden channel / spectral perturbations

---

## Architectural Design

To maximize multi-scale feature extraction while maintaining a lightweight footprint:
- **Compact Parameter Budget:** The model contains only **87,778 trainable parameters**, preventing capacity-driven memorization on small sample sizes.
- **Inception-Style Multi-Branch Modules:**
  - Parallel $1\times 1$, $3\times 3$, and $5\times 5$ convolutions equipped with $1\times 1$ bottleneck channel compression.
  - Parallel $3\times 3$ max-pooling branch with $1\times 1$ projection.
- **Stem & Classification Head:**
  - Convolutional stem ($3 \to 32 \to 64$ channels) with $3\times 3$ kernels.
  - Global Average Pooling (GAP) replacing parameter-heavy dense layers to drastically reduce parameters and eliminate spatial overfitting.
  - Classification head with a compact dense layer (128 units), Dropout ($p=0.5$), and a 10-class linear projection.
- **Optimization & Regularization Recipe:**
  - **Optimizer:** SGD with Nesterov momentum ($0.9$).
  - **Learning Rate Schedule:** Initial LR of $0.05$ with Cosine Annealing over 60 epochs.
  - **Weight Decay:** $5 \times 10^{-4}$ on convolutional and linear weights (biases and BatchNorm scales excluded).
  - **Regularization:** Batch Normalization after every convolution, Dropout ($0.5$), and Early Stopping (patience 15).

---

## Key Experimental Results

All experiments are conducted across **3 fixed random seeds (0, 1, 2)** and reported as **Mean $\pm$ Std**.

### Systematic Ablation Study

| Configuration | Validation Accuracy (Clean) | Clean Test Accuracy | Shifted Test Accuracy | Shifted Gain vs. Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **No augmentation (Baseline)** | $44.7\% \pm 0.6\%$ | $40.8\% \pm 2.1\%$ | $16.5\% \pm 1.6\%$ | Baseline ($+0.0\%$) |
| **Geometric only** (crop, flip, rotation) | $33.3\% \pm 1.2\%$ | $35.3\% \pm 1.4\%$ | $21.9\% \pm 1.4\%$ | **$+5.4\%$** |
| **Photometric only** (color jitter, blur, noise) | $35.3\% \pm 7.5\%$ | $35.1\% \pm 5.5\%$ | $18.8\% \pm 0.7\%$ | $+2.3\%$ |
| **Geometric + photometric** | $28.7\% \pm 2.1\%$ | $28.0\% \pm 0.5\%$ | $20.1\% \pm 2.3\%$ | $+3.6\%$ |
| **+ Cutout / Random Erasing** | $27.3\% \pm 5.1\%$ | $26.7\% \pm 6.0\%$ | $20.8\% \pm 5.0\%$ | $+4.3\%$ |
| **+ CutMix + label smoothing (Top Model)** | $30.3\% \pm 2.5\%$ | $30.3\% \pm 1.7\%$ | **$25.1\% \pm 1.0\%$** | **$+8.6\%$** |

### Insights & Analysis
1. **Winning Configuration:** A composite pipeline combining Geometric transforms (rotation $\pm 25^\circ$, translation $\pm 4$ px, scale $0.9\text{--}1.1$, horizontal flip) + Photometric transforms (brightness, contrast, saturation, blur, noise) + Cutout + CutMix with Label Smoothing ($0.1$). It achieved **$25.1\% \pm 1.0\%$** on the shifted test set.
2. **Single Change Contributing Most:** **Geometric Augmentation** delivered the largest single performance leap ($+5.4$ percentage points, raising shifted accuracy from $16.5\%$ to $21.9\%$). This directly enforces rotational and positional invariance against test corruptions.
3. **Regularization via CutMix:** Adding CutMix and label smoothing delivered the second largest jump ($+4.3$ points from $20.8\%$ to $25.1\%$), mitigating rectangular occlusion and smoothing multi-class decision boundaries.
4. **Baseline Failure Diagnosis:**
   - *Overfitting Gap:* Train accuracy reached $88.9\% \pm 11.3\%$ vs. clean test $40.8\% \pm 2.1\%$ (a $48.1\%$ gap).
   - *Covariate Shift Drop:* Clean test accuracy dropped from $40.8\% \pm 2.1\%$ to $16.5\% \pm 1.6\%$ on shifted test (a $24.3\%$ drop).
   - The unaugmented baseline failed due to both severe overfitting and severe covariate shift.
5. **Test-Time Augmentation (TTA):** Averaging softmax predictions over 8 transformed views (original, flip, 4 directional shifts, 2 flipped shifts) provided stable predictions at an $8.2\times$ inference runtime trade-off.

---

## Repository Structure

```
.
|-- DL_ASS02_23F_0791.ipynb                # Executed Jupyter Notebook with all pipelines, training runs & plots
|-- F230791 Fatima - Assignment#2 DLP Summary.docx # Executive experimental summary report
|-- results.json                           # Raw metric arrays across all seeds and ablation stages
|-- .gitignore                             # Excludes large dataset binaries and cache files
`-- README.md                              # Project documentation and reproduction guide
```

---

## Getting Started & Reproduction

### 1. Requirements & Dependencies
Ensure you have Python 3.9+ installed with the following libraries:
```bash
pip install torch torchvision numpy matplotlib pandas
```
*(Supports CPU as well as CUDA-accelerated GPU environments).*

### 2. Dataset Setup
The notebook uses `torchvision.datasets.CIFAR10(root="./data", download=True)`, which automatically fetches CIFAR-10 upon first run. No manual archive extraction is required.

### 3. Running the Pipeline
1. Clone this repository:
   ```bash
   git clone https://github.com/fatima-Sami55/CIFAR10-Image-Classifier.git
   cd CIFAR10-Image-Classifier
   ```
2. Launch Jupyter Notebook / JupyterLab:
   ```bash
   jupyter notebook DL_ASS02_23F_0791.ipynb
   ```
3. Run all cells sequentially. Fixed seed management guarantees reproducibility:
   - Data split seed: `DATA_SEED = 42`
   - Test corruption seed: `SHIFT_SEED = 123`
   - Validation corruption seed: `VAL_SHIFT_SEED = 7`
   - Experimental runs: `RUN_SEEDS = [0, 1, 2]`

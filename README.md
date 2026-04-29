# Phase Kernel Lifts Capacity of Dense Associative Memory

PyTorch implementation of Phase Kernel Hopfield Networks for dense associative memory.

## Models

| Name | Description |
|------|-------------|
| MHN | Modern Hopfield Networks (baseline) |
| P-Hop | Phase kernel with trigonometric functions |
| S-Hop | Sigmoid-based phase kernel |
| H-Hop | Hoover-style L2-kernel |
| U-Hop | Learnable linear kernel |
| C-Hop | Cosine kernel |

## Requirements

```
Python >= 3.7
PyTorch >= 1.11.0
torchvision >= 0.12.0
numpy / matplotlib / tqdm / scikit-learn / pandas / einops
```

```bash
pip install -r requirements.txt
```

## Repository Structure

```
phase_kernel_code/
├── models.py              # Hopfield attention modules
├── vit_models.py          # Vision Transformer components
├── vih_model.py           # Vision Hopfield (ViH) model
├── functions.py           # Core update rules
├── data.py                # Data loaders (CIFAR-10, MNIST, TinyImageNet)
├── sparse_max.py          # Sparsemax activation
├── entmax.py              # Entmax activation and kernel classes
├── capacity_exp.py        # Memory capacity experiments (max-loss kernel)
├── pooling_models.py      # Memory capacity experiments (avg-loss kernel + PCA)
├── gating_models.py       # Memory capacity experiments (gating mechanism)
├── ssd_models.py          # Memory capacity experiments (SSD metric, U-Hop protocol)
├── classification_exp.py  # Image classification experiments
├── mil_exp.py             # Multiple instance learning experiments
└── tiny_imagenet_exp.py   # TinyImageNet experiments
```

## Usage

```bash
# Memory capacity (max-loss)
python capacity_exp.py

# Memory capacity (avg-loss + PCA)
python pooling_models.py

# Memory capacity (gating)
python gating_models.py

# Memory capacity (SSD metric)
python ssd_models.py

# Image classification on CIFAR-10
python classification_exp.py

# TinyImageNet-200
python tiny_imagenet_exp.py

# Multiple instance learning on MNIST-Bags
python mil_exp.py
```

## Methods

### MHN

Update rule:

$$
x_{t+1} = \Xi \cdot \mathrm{softmax}(\beta \cdot \Xi^\top x_t)
$$

Energy:

$$
E(\xi) = \tfrac{1}{2}\|\xi\|^2 + \tfrac{1}{\beta}\left(\log N + \tfrac{D}{2}\right) - \log\sum_j \exp(\beta \cdot x_j^\top \xi)
$$

---

### P-Hop

Feature map (component-wise for each $k$):

$$
F_P(\phi_k) = \left[\sqrt{\phi_k^2 + \sin\phi_k + 0.5},\ \sqrt{\phi_k^2 + \cos\phi_k + 0.5}\right]
$$

Local field:

$$
h_P(\phi_k) = 2\phi_k + \tfrac{1}{2}(\cos\phi_k - \sin\phi_k)
$$

---

### S-Hop

Feature map (component-wise for each $k$):

$$
F_S(\phi_k) = \left[\sqrt{\phi_k^2 + \tanh\phi_k + 1},\ \sqrt{\phi_k^2 + 1}\right]
$$

Local field:

$$
h_S(\phi_k) = 2\phi_k + \tfrac{1}{2}\,\mathrm{sech}^2(\phi_k)
$$

---

### H-Hop

Based on Hoover et al. (NeurIPS 2024). Uses an L2-distance kernel with pattern-space centering.

Energy:

$$
E_H(x) = -\frac{1}{\beta}\,\mathrm{lse}\left(-\frac{\beta}{2}\|x - \xi_j\|^2\right)
$$

Update rule, where $\Xi_c = \Xi - \bar{\Xi}$ and $x_c = x - \bar{\Xi}$:

$$
x_{t+1} = \Xi \cdot \mathrm{softmax}(\beta \cdot \Xi_c^\top x_c)
$$

---

### U-Hop

Learnable kernel $K(u,v) = (Wu)^\top(Wv)$ with $W \in \mathbb{R}^{d\times d}$ trained via uniformity loss:

$$
\mathcal{L}_{\mathrm{uniform}}(X) = \log\left(\mathrm{mean}_{i \neq j}\,\exp(-t\|x_i - x_j\|^2)\right)
$$

Update rule:

$$
x_{t+1} = \Xi \cdot \mathrm{softmax}\left(\beta \cdot (W\Xi)^\top(Wx)\right)
$$

---

### C-Hop

Cosine-sine embedding applied after $[\min,\max]$ rescaling with scaling factor $n=100$:

$$
\varphi_{\mathrm{scaled}} = \frac{\varphi - \min(\varphi)}{\max(\varphi) - \min(\varphi)} \cdot n, \qquad F_C(\varphi) = [\cos\varphi,\ \sin\varphi]
$$

---

## Capacity Experiment Variants

**Avg-Loss (`pooling_models.py`)** — Uses mean uniform loss (instead of max) for U-Hop kernel training, plus PCA denoising. Otherwise identical to `capacity_exp.py`.

**Gating (`gating_models.py`)** — Wraps standard update rules with a learned gate:

$$
g = \sigma(\gamma \cdot z), \qquad x_{t+1} = (1-g)\,x_t + g\,x_{\mathrm{prop}}
$$

$$
z_{t+1} = (1-\lambda)\,z_t + \lambda\,\phi(Wx_t)
$$

Applied to MHN, U-Hop, S-Hop; C-Hop is tested without gating as a reference.

**SSD Metric (`ssd_models.py`)** — Follows the U-Hop official SSM protocol. Metric:

$$
\mathrm{SSD}(x,\hat{x}) = \sum_{i=1}^{d}\left(\mathrm{clamp}(x_i,0,1) - \mathrm{clamp}(\hat{x}_i,0,1)\right)^2
$$

N range: $[10, 20, 30, 50, 100, 200, 500]$. Query: Gaussian noise or dropout mask.

## Hyperparameters

**Memory Capacity** (`capacity_exp.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 1.0 | Inverse temperature |
| `steps` | 1 | Update iterations |
| `noise_level` | 0.3 | Query noise std |
| `perfect_threshold` | 0.10 | MSE threshold for perfect retrieval |
| `capacity_threshold` | 0.90 | Perfect retrieval rate threshold |

**Classification** (`classification_exp.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 128 | Training batch size |
| `init_lr` | 3e-4 | Initial learning rate |
| `epoch` | 100 | Training epochs |
| `depth` | 6 | Number of layers |
| `update_steps` | 3 | Hopfield iterations per layer |
| `num_iter` | 10 | Newton iterations for S-Hop |

**TinyImageNet** (`tiny_imagenet_exp.py`)

| Parameter | Default |
|-----------|---------|
| `batch_size` | 128 |
| `epochs` | 100 |
| `emb_size` | 384 |
| `patch_size` | 4 |
| `n_heads` | 6 |

**MIL** (`mil_exp.py`)

| Parameter | Default |
|-----------|---------|
| `target_number` | 9 |
| `num_bag` | 2000 train / 500 test |
| `hidden_dim` | 256 |
| `depth` | 3 |

## Implementation Notes

**Pattern-space centering** — H-Hop centers both stored patterns and the query: $\Xi_c = \Xi - \bar{\Xi}$, $x_c = x - \bar{\Xi}$. Other methods operate in the original feature space.

**Newton iterations** — P-Hop and S-Hop invert the local field $h^{-1}$ via Newton's method. Default: 20 iterations for capacity experiments, 10 for classification.

**Jacobian coupling** — P-Hop and S-Hop include Jacobian terms in the "values" tensor to ensure correct gradient flow through the phase kernel.

## Datasets

| Dataset | Spec |
|---------|------|
| CIFAR-10 | 32×32 RGB, 10 classes, 50k train / 10k test |
| MNIST | 28×28 grayscale, 10 digits, 60k train / 10k test |
| TinyImageNet-200 | 64×64 RGB, 200 classes, 100k train / 10k val |

TinyImageNet is downloaded automatically on first run.

## Citation

If you use this code, please cite our paper.


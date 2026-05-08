# EEGConformer — Model Architecture

## Overview

EEGConformer combines a CNN patch embedding stage with a Transformer encoder for cross-subject EEG classification (error-related potential detection). Input is a single EEG epoch; output is a raw logit for binary classification (correct / error).

- **Input:** `(B, 56, 260)` — 56 EEG channels × 260 timepoints (1.3 s @ 200 Hz)
- **Output:** `(B, 1)` — raw logit (apply sigmoid for probability)
- **Preprocessing:** Per-subject Euclidean Alignment (EA) must be applied before inference

---

## Architecture Diagram

![EEGConformer Architecture](figures/architecture.png)

---

## Parameter Count

### Patch Embedding

| Sub-layer | Parameters |
|---|---:|
| Temporal Conv2d (1 → 64, kernel 1×25) + bias | 1,664 |
| Temporal BatchNorm2d (64) | 128 |
| Spatial DepthwiseConv2d (64 → 64, kernel 56×1) + bias | 3,648 |
| Spatial BatchNorm2d (64) | 128 |
| **Patch Embedding total** | **5,568** |

### Positional Encoding

| Component | Parameters |
|---|---:|
| Learnable pos_enc (1 × 13 × 64) | 832 |

### Transformer Encoder (4 blocks, identical)

| Sub-layer (per block) | Parameters |
|---|---:|
| Multi-Head Self-Attention (in/out proj, 8 heads) | 16,640 |
| Feed-Forward Network (64 → 256 → 64) | 33,088 |
| LayerNorm × 2 | 256 |
| **Per-block total** | **49,984** |
| **4 blocks total** | **199,936** |

### Classifier Head

| Sub-layer | Parameters |
|---|---:|
| Final LayerNorm (64) | 128 |
| Linear (832 → 256) + bias | 213,248 |
| Linear (256 → 1) + bias | 257 |
| **Head total** | **213,633** |

### Summary

| Module | Parameters | % of total |
|---|---:|---:|
| Patch Embedding | 5,568 | 1.3% |
| Positional Encoding | 832 | 0.2% |
| Transformer Encoder (×4) | 199,936 | 47.6% |
| Classifier Head (LayerNorm + Linear×2) | 213,633 | 50.9% |
| **Total** | **419,969** | **100%** |

All 419,969 parameters are trainable.

---

## Shapes Through the Network

```
Input             (B, 56, 260)
Temporal Conv2d   (B, 64, 56, 13)   ← AvgPool reduces T: floor((260−75)/15)+1 = 13
Spatial DWConv    (B, 64, 1,  13)
Reshape + permute (B, 13, 64)        ← sequence of 13 tokens, dim=64
+ Positional Enc  (B, 13, 64)
Transformer ×4    (B, 13, 64)
LayerNorm         (B, 13, 64)
Flatten           (B, 832)           ← 13 × 64
Linear + GELU     (B, 256)
Linear            (B, 1)             ← raw logit
```

---

## Cross-Validation Results

Trained with subject-wise 4-fold CV, 3 repeats (12 checkpoints total). Euclidean Alignment applied per subject before training and inference.

| Config | CV AUC |
|---|---:|
| Per-fold mean | 0.7192 ± 0.0405 |
| **Ensemble (all 12 checkpoints)** | **0.7305** |
| Winner baseline | 0.7294 |

Ensemble beats the Riemannian + ElasticNet + 500-bag baseline.

### Per-checkpoint AUC

| Checkpoint | Val AUC |
|---|---:|
| fold_rep1_fold1.pt | **0.7760** |
| fold_rep2_fold1.pt | 0.7710 |
| fold_rep0_fold0.pt | 0.7545 |
| fold_rep2_fold3.pt | 0.7453 |
| fold_rep1_fold0.pt | 0.7142 |
| fold_rep1_fold2.pt | 0.7111 |
| fold_rep1_fold3.pt | 0.7053 |
| fold_rep0_fold3.pt | 0.7032 |
| fold_rep2_fold2.pt | 0.6957 |
| fold_rep2_fold0.pt | 0.6871 |
| fold_rep0_fold1.pt | 0.6891 |
| fold_rep0_fold2.pt | 0.6775 |

Best single checkpoint: `checkpoints/fold_rep1_fold1.pt`

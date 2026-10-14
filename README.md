# Fully Convolutional Networks (FCN) for Semantic Segmentation
**Deep Learning 2026 — Project DL-029**  
**Student:** Roll No: `24CS2038` | Section: `IDD`  
**Paper:** Jonathan Long, Evan Shelhamer, Trevor Darrell. *"Fully Convolutional Networks for Semantic Segmentation."* CVPR 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)

---

## Overview

This repository contains my PyTorch implementation and reproduction of the FCN paper for the Deep Learning course. 

The goal of this project is to implement the core ideas of the CVPR 2015 paper:
1. Converting standard classification CNNs (VGG-16) into fully convolutional networks by turning fully connected layers into convolutions.
2. In-network upsampling using fractionally strided (transposed) convolutions initialized with 2D bilinear interpolation weights.
3. Building the multi-scale skip architecture (FCN-32s, FCN-16s, and FCN-8s) and evaluating the effect of fusing shallow features with deep features.
4. Comparing quantitative results (mIoU, pixel accuracy) and qualitative boundary quality across the models.

```
Input Image (H, W)
  │
  ├─► [Stage 1-3] ──► Pool3 (stride 8, 256 ch) ────────┐ (1x1 conv)
  │                                                     │
  ├─► [Stage 4]   ──► Pool4 (stride 16, 512 ch) ──┐     │
  │                                               │     │
  └─► [Stage 5 + Conv6/7] ──► Conv7 (stride 32)   │     │
                                 │                │     │
                         [Score 1x1 conv]         │     │
                                 │                │     │
                        ┌────────┴────────┐       │     │
                        ▼                 ▼       │     │
                    [32x Up]          [2x Up]     │     │
                        │                 │       │     │
                    (FCN-32s)             └─(+)───┘     │
                                             │          │
                                     ┌───────┴──────┐   │
                                     ▼              ▼   │
                                 [16x Up]        [2x Up]│
                                     │              │   │
                                 (FCN-16s)          └─(+)
                                                      │
                                                   [8x Up]
                                                      │
                                                  (FCN-8s)
```

---

## Directory Structure

```
├── models/
│   ├── __init__.py           # Model factory
│   └── fcn.py                # VGGBackbone, FCN-32s, FCN-16s, FCN-8s, bilinear weights
├── utils/
│   ├── dataset.py            # CamVid dataloader, color palette, augmentations
│   ├── metrics.py            # Evaluator: Pixel Acc, Mean Acc, mIoU, FW-IoU
│   └── visualization.py      # Comparison plots and training curves
├── report/
│   ├── report.tex            # 5-page IEEE two-column paper (LaTeX)
│   └── REPORT.md             # Markdown version of report
├── train.py                  # Training script
├── evaluate.py               # Evaluation on validation set
├── run_ablation.py           # Automated comparison between 32s, 16s, and 8s
├── FCN_Semantic_Segmentation_24CS2038.ipynb # Runnable Google Colab notebook
├── PRESENTATION_AND_VIVA.md  # Viva presentation guide & board derivations
├── requirements.txt
└── README.md
```

---

## Setup & Dependencies

Python 3.10+ with standard deep learning libraries:
```bash
pip install -r requirements.txt
```

Packages used: `torch`, `torchvision`, `numpy`, `pillow`, `matplotlib`.

---

## How to Run

### 1. Train FCN-8s
```bash
python train.py --model fcn8s --dataset camvid --epochs 20 --batch-size 4 --lr 1e-4
```

To train FCN-32s or FCN-16s instead:
```bash
python train.py --model fcn32s --epochs 20
python train.py --model fcn16s --epochs 20
```

To test random weight initialization instead of bilinear initialization (ablation study):
```bash
python train.py --model fcn8s --no-bilinear
```

### 2. Evaluate a Saved Checkpoint
```bash
python evaluate.py --model fcn8s --checkpoint ./checkpoints/fcn8s_best.pth
```

### 3. Run Full Ablation Pipeline
This trains FCN-32s, FCN-16s, and FCN-8s under identical random seeds (`seed=42`) and generates the comparison plots and tables:
```bash
python run_ablation.py --epochs 10 --output-dir ./results
```

### 4. Run on Google Colab
You can run the entire workflow in Google Colab using `FCN_Semantic_Segmentation_24CS2038.ipynb`. It is self-contained and runs on a free T4 GPU instance.

---

## Reproduction Results

Trained on CamVid road scene dataset (11 classes + void label 255):

| Architecture | Stride | Pixel Accuracy | Mean Accuracy | mIoU | FW-IoU | Paper VOC mIoU Trend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FCN-32s** (No skip) | 32 px | 82.41% | 61.15% | **58.24%** | 74.32% | 59.4% (Baseline) |
| **FCN-16s** (Pool4 skip) | 16 px | 86.73% | 68.42% | **64.12%** | 79.85% | 62.4% (+3.0%) |
| **FCN-8s** (Pool4 + Pool3) | 8 px | **88.95%** | **71.30%** | **66.83%** | **82.40%** | 62.7% (+0.3%) |

### Observations:
1. **Adding Pool4 Skip (FCN-16s):** Gives the biggest jump in mIoU (+5.88% in our run, +3.0% in the paper). Bringing stride-16 features helps define rough object outlines that were lost at stride 32.
2. **Adding Pool3 Skip (FCN-8s):** Provides further refinement (+2.71% mIoU). Looking at per-class numbers, this primarily helps small and thin classes:
   - *Poles:* 21.4% (32s) -> 38.9% (8s) (+17.5% gain)
   - *Pedestrians:* 42.1% (32s) -> 55.8% (8s) (+13.7% gain)
   - *Signs:* 28.6% (32s) -> 44.2% (8s) (+15.6% gain)
   Large uniform classes (Road, Sky) were already >90% in FCN-32s, so skips had minimal impact there.

---

## Ablation: Deconvolution Initialization

We compared 2D bilinear initialization vs standard random initialization (He/Kaiming normal) in FCN-8s:

| Initialization | Initial Loss | Epoch 3 mIoU | Final mIoU | Note |
| :--- | :---: | :---: | :---: | :--- |
| **Random (He normal)** | 3.48 | 46.1% | 61.3% | Slower convergence, noisy early gradients |
| **Bilinear Filter** | **2.12** | **59.4%** | **66.8%** | Starts from smooth interpolation, +5.5% final gain |

Bilinear initialization acts as a strong warm-start, allowing the network to focus on learning boundary corrections rather than learning how to scale up spatial resolution from scratch.

---

## Implementation Details & Engineering Challenges

1. **VGG-16 Classifier Reshaping:**
   In PyTorch's `vgg16`, `classifier[0]` has shape `(4096, 25088)`. Because $25088 = 512 \times 7 \times 7$, we reshaped this tensor to `(4096, 512, 7, 7)` to initialize our `conv6` layer. Similarly, `classifier[3]` with shape `(4096, 4096)` was viewed as `(4096, 4096, 1, 1)` for `conv7`.
2. **Transposed Convolution Padding:**
   Setting padding wrong caused off-by-one spatial dimension mismatches. For an upsampling factor of 32 with kernel size 64 and stride 32, the exact formula is:
   $$H_{out} = (H_{in} - 1) \cdot 32 - 2(16) + 64 = 32 \cdot H_{in}$$
   which requires `padding=16`.
3. **GPU Memory:**
   Due to VGG-16 having 4096 feature channels in `conv6` and `conv7`, intermediate activations are memory heavy. On a Colab T4 (16GB), batch size 4 ran comfortably without out-of-memory errors.

---

## Deliverables

- **IEEE Report:** Located in [`report/report.tex`](./report/report.tex) and rendered in [`report/REPORT.md`](./report/REPORT.md).
- **Colab Notebook:** [`FCN_Semantic_Segmentation_24CS2038.ipynb`](./FCN_Semantic_Segmentation_24CS2038.ipynb).
- **Presentation & Viva Prep:** Detailed slide outline and derivations in [`PRESENTATION_AND_VIVA.md`](./PRESENTATION_AND_VIVA.md).

---

## Citation
```bibtex
@inproceedings{long2015fully,
  title={Fully convolutional networks for semantic segmentation},
  author={Long, Jonathan and Shelhamer, Evan and Darrell, Trevor},
  booktitle={CVPR},
  pages={3431--3440},
  year={2015}
}
```

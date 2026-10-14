# Implementation and Multi-Scale Skip Architecture Ablation of Fully Convolutional Networks (FCN)

**Course:** Deep Learning 2026 (Paper ID: DL-029)  
**Student:** Roll No: `24CS2038` | Section: `IDD`  
**Reference Paper:** Jonathan Long, Evan Shelhamer, Trevor Darrell. *"Fully Convolutional Networks for Semantic Segmentation."* IEEE CVPR 2015.

---

## 1. Problem Statement & Motivation

Semantic segmentation involves predicting a categorical label for every individual pixel in an image:

$$\mathbf{Y} \in \{1, 2, \dots, C\}^{H \times W}$$

Before the FCN paper (Long et al., 2015), deep learning methods for segmentation relied on patch-based approaches. Small patches centered around each pixel were cropped and passed through a classification network. This was computationally slow due to repeated convolutions across overlapping patches, and it forced a trade-off:
- Small patches preserve edge details but lack context.
- Large patches provide semantic context but blur boundaries.

Furthermore, standard classification networks (such as AlexNet or VGG-16) downsample features by $32\times$ through consecutive max-pooling layers, and their terminal fully connected layers discard all 2D spatial arrangements to output a 1D vector of class probabilities.

### Long et al.'s Solution:
1. **Convolutionalization:** A fully connected layer can be viewed as a convolution whose filter covers the entire incoming spatial dimensions. Converting FC layers into convolutions allows the network to accept images of **any arbitrary resolution** and output a 2D spatial heatmap in a single forward pass.
2. **In-Network Upsampling:** Using fractionally strided (transposed) convolutions initialized with 2D bilinear interpolation weights to upsample coarse feature maps back to the input resolution.
3. **Multi-Scale Skip Connections:** Combining deep semantic representations from `conv7` with shallow, high-resolution appearance features from `pool4` and `pool3` to restore fine boundary details.

---

## 2. Methodology & Mathematical Details

### 2.1 Convolutionalizing the VGG-16 Backbone
In standard VGG-16, the output of `pool5` has shape $512 \times 7 \times 7$. The first fully connected layer (`fc6`) flattens this into $25,088$ values and connects them to $4096$ units:

$$y_k = \sum_{c=1}^{512} \sum_{u=0}^{6} \sum_{v=0}^{6} W_{k, c, u, v} \cdot x_{c, u, v} + b_k$$

In FCN, this layer is converted into a 2D convolution:
- Kernel size: $7 \times 7$
- In channels: $512$
- Out channels: $4096$
- Padding: $3$ (preserving spatial dimensions at $H/32 \times W/32$)

The pretrained weights from PyTorch's `vgg16` are directly reshaped:

$$\mathbf{K}_{\text{conv6}} = \text{reshape}(\mathbf{W}_{\text{fc6}}, (4096, 512, 7, 7))$$

Similarly, `fc7` ($4096 \rightarrow 4096$) is converted into a $1 \times 1$ convolution:

$$\mathbf{K}_{\text{conv7}} = \text{reshape}(\mathbf{W}_{\text{fc7}}, (4096, 4096, 1, 1))$$

### 2.2 Transposed Convolutions
To upsample the downsampled feature maps back to full input resolution, FCN uses transposed convolution layers (`nn.ConvTranspose2d`).

The output spatial dimension of a transposed convolution with input dimension $H_{in}$, stride $s$, kernel size $k$, and padding $p$ is:

$$H_{out} = (H_{in} - 1) \cdot s - 2p + k$$

For exact $32\times$ upsampling with stride $s = 32$:
- Set $k = 64$ and $p = 16$:
  $$H_{out} = (H_{in} - 1) \cdot 32 - 2(16) + 64 = 32 \cdot H_{in}$$

### 2.3 2D Bilinear Weight Initialization
Rather than initializing deconvolution layers randomly, Long et al. initialized them with 2D bilinear interpolation weights:

$$w_{i, j, u, v} = \delta_{ij} \left(1 - \frac{|u - c|}{f_s}\right) \left(1 - \frac{|v - c|}{f_s}\right)$$

where $f_s = \lceil k / 2 \rceil$ is the scale factor, $c = (k + 1) / 2 - 1$ is the kernel center, and $\delta_{ij}$ ensures that each class channel is upsampled independently. This allows the network to begin from a clean bilinear upsampling baseline and learn non-linear edge refinements through standard gradient descent.

### 2.4 Multi-Scale Skip Architecture Hierarchy
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

1. **FCN-32s:** Predicts class scores directly from `conv7` and upsamples $32\times$.
2. **FCN-16s:** Upsamples `conv7` scores by $2\times$ and sums them with scores from `pool4` (stride 16), then upsamples $16\times$.
3. **FCN-8s:** Upsamples the FCN-16s fused scores by $2\times$ and sums them with scores from `pool3` (stride 8), then upsamples $8\times$.

---

## 3. Experimental Setup

- **Dataset:** CamVid road scene dataset with 11 semantic classes (`Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist`) and void label 255.
- **Image Size:** $352 \times 480$ pixels.
- **Backbone:** VGG-16 pretrained on ImageNet, with converted `conv6` and `conv7`.
- **Optimizer:** SGD with momentum 0.9, weight decay $5 \times 10^{-4}$, initial learning rate $10^{-4}$ with cosine annealing.
- **Batch Size:** 4 (constrained by GPU memory due to VGG's 4096-channel intermediate feature maps).
- **Seed:** Fixed at `42` for reproducibility across all runs.

### Evaluation Metrics (Section 5 of paper)
From the confusion matrix $N \in \mathbb{R}^{C \times C}$ where $n_{ij}$ is the count of class $i$ pixels predicted as class $j$, and $t_i = \sum_j n_{ij}$:
1. **Pixel Accuracy:** $\frac{\sum_i n_{ii}}{\sum_i t_i}$
2. **Mean Accuracy:** $\frac{1}{C} \sum_i \frac{n_{ii}}{t_i}$
3. **Mean IoU (mIoU):** $\frac{1}{C} \sum_i \frac{n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}$
4. **Frequency-Weighted IoU:** $\frac{1}{\sum t_k} \sum_i \frac{t_i \cdot n_{ii}}{t_i + \sum_j n_{ji} - n_{ii}}$

---

## 4. Quantitative Results & Reproduction

### Table 1: Reproduction of Multi-Scale Skip Progression
| Model | Stride | Pixel Accuracy | Mean Accuracy | mIoU | FW-IoU | Paper VOC Trend |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FCN-32s** (No skip) | 32 px | 82.41% | 61.15% | **58.24%** | 74.32% | 59.4% (Baseline) |
| **FCN-16s** (Pool4 skip) | 16 px | 86.73% | 68.42% | **64.12%** | 79.85% | 62.4% (+3.0%) |
| **FCN-8s** (Pool4 + Pool3) | 8 px | **88.95%** | **71.30%** | **66.83%** | **82.40%** | 62.7% (+0.3%) |

### Observations:
- **Largest improvement at FCN-16s:** Adding `pool4` features increases mIoU by **+5.88%** (compared to +3.0% on VOC in the original paper). Stride-16 features provide the necessary spatial grounding that stride-32 features blur out.
- **FCN-8s refinement:** Adding `pool3` features provides an additional **+2.71%** boost in mIoU, sharpening thin structures and borders.

### Table 2: Per-Class Breakdown (Why Skip Connections Help)
| Class | FCN-32s IoU (%) | FCN-16s IoU (%) | FCN-8s IoU (%) | Gain (8s vs 32s) |
| :--- | :---: | :---: | :---: | :---: |
| **Road** | 92.1 | 94.5 | 95.7 | +3.6% |
| **Building** | 78.5 | 82.1 | 84.3 | +5.8% |
| **Car** | 74.8 | 80.2 | 83.4 | +8.6% |
| **Pedestrian** | 42.1 | 51.3 | 55.8 | **+13.7%** |
| **Pole** | 21.4 | 31.7 | 38.9 | **+17.5%** |
| **SignSymbol** | 28.6 | 39.1 | 44.2 | **+15.6%** |

*Key finding:* Large regions (like Road and Sky) already perform well in FCN-32s (>91%). The major benefits of skip connections come from thin, small classes (Poles: +17.5%, Signs: +15.6%, Pedestrians: +13.7%), which require high-resolution features from earlier pooling stages.

---

## 5. Ablation: Transposed Convolution Initialization

We evaluated the impact of initializing transposed convolutions with 2D bilinear filters versus standard random Gaussian (He normal) weights in FCN-8s:

| Initialization | Initial Loss | Epoch 3 mIoU | Final mIoU (%) |
| :--- | :---: | :---: | :---: |
| **Random (He normal)** | 3.48 | 46.10% | 61.32% |
| **Bilinear Filter** | **2.12** | **59.40%** | **66.83%** |

Bilinear initialization provides a strong structural warm-start. Instead of spending early epochs learning how to interpolate pixels, the network starts from smooth spatial upsampling and immediately begins refining boundary decisions.

---

## 6. Implementation Challenges & Engineering Notes

1. **Weight Transfer from VGG-16:**
   PyTorch's `vgg16` stores `classifier[0]` as a flat matrix of shape `[4096, 25088]`. Because $25088 = 512 \times 7 \times 7$, we reshaped it into `[4096, 512, 7, 7]` to properly initialize `conv6`.
2. **Padding in Deconvolution Layers:**
   Early experiments had off-by-one spatial dimension errors. Using the equation $H_{out} = (H_{in} - 1)s - 2p + k$, setting stride $s = 32$, kernel $k = 64$, and padding $p = 16$ cleanly achieves exact $32\times$ upsampling.
3. **GPU Memory:**
   Due to VGG's wide layers (4096 channels at stride 32), batch sizes larger than 4 caused CUDA Out of Memory errors on standard 16GB GPUs. Using batch size 4 with cosine learning rate decay converged smoothly.

---

## 7. Failure Cases

Inspection of predicted validation masks showed three recurring failure patterns:
1. **Thin and Fragmented Objects:** Very fine structures (such as bicycle frames or thin utility lines) vanish during max-pooling and cannot be recovered by transposed convolution.
2. **Boundary Smoothing:** Because FCN-8s still performs an $8\times$ upsampling step at the end, predicted object boundaries remain slightly rounded compared to ground truth.
3. **Internal Hole Artifacts on Large Vehicles:** Large trucks or buses close to the camera occasionally have small misclassified regions in their center due to receptive field limitations.

---

## 8. Conclusion
We successfully implemented FCN-32s, FCN-16s, and FCN-8s from first principles in PyTorch. The empirical results verify that convolutionalizing classification networks allows dense pixel prediction, skip connections from shallow layers are critical for recovering fine boundary structures, and 2D bilinear initialization accelerates convergence and boosts final mIoU.

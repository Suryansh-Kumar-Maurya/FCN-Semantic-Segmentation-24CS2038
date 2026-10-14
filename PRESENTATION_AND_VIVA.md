# Presentation & Viva Preparation Guide
**Deep Learning 2026 — Project DL-029**  
**Student:** Roll No: `24CS2038` | Section: `IDD`  
**Paper:** Fully Convolutional Networks for Semantic Segmentation (Long, Shelhamer, Darrell — CVPR 2015)  
**Time Limit:** ~10 minutes presentation + Q&A / Blackboard Derivations

---

## 1. 10-Minute Slide-by-Slide Speaking Notes

### Slide 1: Introduction (0:00 – 0:45)
- **Title:** Fully Convolutional Networks for Semantic Segmentation
- **Details:** Roll No: `24CS2038`, Section: `IDD`, Paper ID: `DL-029`
- **What to say:**
  > *"Hello professors, my roll number is 24CS2038 from the IDD section. For my Deep Learning paper reproduction project, I implemented the 2015 CVPR paper on Fully Convolutional Networks by Jonathan Long, Evan Shelhamer, and Trevor Darrell. My presentation covers the convolutionalization of the VGG backbone, the implementation of in-network transposed convolutions, and the ablation of skip connections across FCN-32s, 16s, and 8s."*

---

### Slide 2: Problem & Motivation (0:45 – 1:45)
- **The Problem:** Semantic segmentation requires assigning a class label to every pixel in an image ($H \times W$).
- **Prior Approach (Patch-based):** Cropping small local patches around every pixel and passing them through a CNN.
  - Slow and repetitive: overlapping patches compute the same convolutions dozens of times.
  - Patch-size dilemma: small patches capture edges but lack context; large patches capture context but blur edges.
- **The Core Issue in Standard CNNs:** Classification networks (AlexNet, VGG) pool feature maps down by $32\times$ and end with fully connected layers that discard spatial layout into a 1D vector.

---

### Slide 3: Convolutionalization (1:45 – 3:00)
- **Key Insight:** A fully connected layer is just a convolution whose kernel size equals the spatial size of the input feature map.
- **How it works:**
  - VGG-16 `pool5` outputs $512 \times 7 \times 7$ feature maps.
  - `fc6` connects all $512 \times 7 \times 7 = 25,088$ units to 4096 units.
  - By replacing `fc6` with a $7 \times 7$ convolution (padding 3), we keep the exact same receptive field while preserving spatial coordinates.
  - `fc7` ($4096 \rightarrow 4096$) becomes a $1 \times 1$ convolution.
- **Weight Transfer:** The pretrained weights from `vgg16.classifier[0]` of shape `[4096, 25088]` are reshaped to `[4096, 512, 7, 7]` and copied directly into `conv6`.
- **Benefit:** The network now takes inputs of **any arbitrary resolution** and computes dense spatial predictions in a single forward pass.

---

### Slide 4: In-Network Transposed Convolutions & Bilinear Init (3:00 – 4:15)
- **How to Upsample:** To scale stride-32 features back to full resolution, we use transposed convolutions (`nn.ConvTranspose2d`).
- **Spatial Formula:** $H_{out} = (H_{in} - 1) \cdot s - 2p + k$.
  For $s = 32, k = 64, p = 16$: $H_{out} = (H_{in} - 1) \cdot 32 - 32 + 64 = 32 \cdot H_{in}$ (exact $32\times$ upsample).
- **Bilinear Initialization:** Rather than initializing deconvolution weights randomly, Long et al. initialize them with 2D bilinear interpolation weights.
- **Why it matters:** It warm-starts the layer so it acts like smooth bilinear upsampling at epoch 0, allowing the network to focus on learning non-linear boundary adjustments rather than learning upsampling from scratch.

---

### Slide 5: Multi-Scale Skip Architectures (4:15 – 5:30)
- **FCN-32s:** Direct $32\times$ upsample from `conv7`. Coarse and blobby because stride 32 loses high-frequency details.
- **FCN-16s:** Upsample `conv7` scores by $2\times$ (to stride 16), add to `pool4` scores (projected with a $1 \times 1$ conv), then upsample $16\times$.
- **FCN-8s:** Upsample FCN-16s fused scores by $2\times$ (to stride 8), add to `pool3` scores, then upsample $8\times$.
- **Core Principle:** Deep layers know *what* is in the image (high semantics), but shallow layers know *where* it is (exact pixel coordinates). Adding them combines both.

---

### Slide 6: Experimental Setup (5:30 – 6:15)
- **Dataset:** CamVid road scene dataset (11 classes + void index 255).
- **Image Resolution:** $352 \times 480$.
- **Training:** SGD with momentum 0.9, weight decay $5 \times 10^{-4}$, initial learning rate $10^{-4}$ with cosine decay.
- **Hardware/Memory:** Batch size 4. VGG-16 activations with 4096 channels at stride 32 are memory intensive, so batch size 4 fits comfortably in standard 16GB GPU memory without OOM.
- **Metrics Evaluated:** Pixel Accuracy, Mean Accuracy, Mean IoU (mIoU), and Frequency-Weighted IoU.

---

### Slide 7: Quantitative Results vs. Paper (6:15 – 7:30)
- **Our Numbers:**
  - **FCN-32s:** 82.41% Pixel Acc | **58.24%** mIoU
  - **FCN-16s:** 86.73% Pixel Acc | **64.12%** mIoU (**+5.88% jump**)
  - **FCN-8s:** **88.95% Pixel Acc** | **66.83%** mIoU (**+2.71% jump**)
- **Comparison to Paper:**
  - Long et al. reported a +3.0% jump from 32s to 16s on VOC, and +0.3% from 16s to 8s.
  - Our reproduction confirms that the first skip connection (`pool4`) provides the biggest single gain in segmentation accuracy.

---

### Slide 8: Per-Class Analysis (7:30 – 8:30)
- **Where did the gains come from?**
  - **Large classes (Road, Sky):** FCN-32s already reached $>91\%$ IoU because coarse features are enough to recognize the ground and sky.
  - **Small/Thin classes (Poles, Signs, Pedestrians):** FCN-32s failed badly (Poles: 21.4%, Signs: 28.6%). FCN-8s recovered them significantly:
    - *Poles:* $21.4\% \rightarrow 38.9\%$ (+17.5%)
    - *Signs:* $28.6\% \rightarrow 44.2\%$ (+15.6%)
    - *Pedestrians:* $42.1\% \rightarrow 55.8\%$ (+13.7%)
- **Takeaway:** Skip connections are essential specifically for high-frequency spatial structures that get wiped out by 5 pooling stages.

---

### Slide 9: Ablation Study & Failure Cases (8:30 – 9:30)
- **Initialization Ablation:** Bilinear initialized FCN-8s reached **66.83%** mIoU vs. **61.32%** for random Gaussian weights (+5.5% advantage, converged in half the epochs).
- **Observed Failure Cases:**
  1. *Very thin structures:* Bicycle spokes and thin wires disappear during pooling and cannot be recovered.
  2. *Rounded boundaries:* Even FCN-8s still upsamples by $8\times$, leading to slightly smoothed corners.
  3. *Internal holes:* Large buses/trucks occasionally have small misclassified patches inside them due to local receptive field limitations.

---

### Slide 10: Conclusion & Submitted Artifacts (9:30 – 10:00)
- **Summary:** Verified convolutionalization, demonstrated multi-scale skip architecture benefits, confirmed bilinear initialization advantage.
- **Code & Report:** Modular PyTorch codebase, runnable Google Colab notebook, and 5-page IEEE report submitted.
- **Closing:** *"Thank you. I am ready to derive equations on the board or answer any questions."*

---

## 2. Blackboard Mathematical Derivations

### Derivation 1: Transposed Convolution as Matrix Transpose
If the examiner asks you to show why it is called "transposed" convolution on the board:
1. Write down a 1D convolution of an input vector $\mathbf{x} = [x_1, x_2, x_3, x_4]^T$ with a kernel $\mathbf{w} = [w_1, w_2, w_3]^T$, stride 1, no padding:
   $$\mathbf{y} = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \begin{bmatrix} w_1 & w_2 & w_3 & 0 \\ 0 & w_1 & w_2 & w_3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} = \mathbf{C} \mathbf{x}$$
   Here $\mathbf{C} \in \mathbb{R}^{2 \times 4}$. Output size is 2.
2. In backpropagation, the gradient with respect to the input is:
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \mathbf{C}^T \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$$
   where $\mathbf{C}^T \in \mathbb{R}^{4 \times 2}$:
   $$\mathbf{C}^T = \begin{bmatrix} w_1 & 0 \\ w_2 & w_1 \\ w_3 & w_2 \\ 0 & w_3 \end{bmatrix}$$
3. A **transposed convolution** simply uses the matrix transpose $\mathbf{C}^T$ in the **forward pass**:
   $$\mathbf{x}_{\text{up}} = \mathbf{C}^T \mathbf{y}$$
   This maps a smaller 2D vector back to a larger 4D vector. Because the forward pass uses the transpose of the convolution matrix $\mathbf{C}$, it is called transposed convolution.

---

### Derivation 2: Output Spatial Dimension Equation
1. In standard 2D convolution:
   $$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - k}{s} \right\rfloor + 1$$
2. Transposed convolution inverts this operation:
   $$(H_{\text{conv}} - 1) \cdot s = H_{\text{in}} + 2p - k$$
   Solving for output spatial dimension $H_{\text{out}}$:
   $$\mathbf{H_{\text{out}} = (H_{\text{in}} - 1) \cdot s - 2p + k}$$
3. **Show exact $32\times$ upsampling:**
   For $s = 32$, $k = 64$, $p = 16$:
   $$H_{\text{out}} = (H_{\text{in}} - 1) \cdot 32 - 2(16) + 64$$
   $$H_{\text{out}} = 32 H_{\text{in}} - 32 - 32 + 64 = \mathbf{32 \cdot H_{\text{in}}}$$
   This proves that with $k=64$ and $p=16$, the output is an exact $32\times$ multiple of the input.

---

### Derivation 3: 2D Bilinear Kernel Equation
1. Continuous 1D linear interpolation triangle filter with scale factor $f_s = \lceil k/2 \rceil$ centered at $c$:
   $$h(u) = 1 - \frac{|u - c|}{f_s}$$
2. Because 2D bilinear interpolation is spatially separable:
   $$w(u, v) = h(u) \cdot h(v) = \left(1 - \frac{|u - c|}{f_s}\right) \left(1 - \frac{|v - c|}{f_s}\right)$$
3. In `get_upsampling_weight`, we set:
   $$W_{i, j, u, v} = \begin{cases} w(u, v) & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$
   This ensures each class channel is upsampled independently without mixing channels before training starts.

---

## 3. Line-by-Line Code Explanation (`models/fcn.py`)

If asked to explain your code:

1. **`get_upsampling_weight` (lines 14–33):**
   - `factor = (kernel_size + 1) // 2`: computes the upsampling scale factor.
   - `og = np.ogrid[:kernel_size, :kernel_size]`: generates 2D coordinate grid arrays for vectorized calculation.
   - `weight[i, j, :, :] = filt` for `i == j`: diagonal weight assignment so each channel scales independently.

2. **`VGGBackbone` (lines 36–85):**
   - `self.stage1` through `self.stage5`: breaks down VGG-16's feature layers at each max-pooling boundary.
   - `self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)`: replaces `fc6`. Padding 3 keeps spatial shape at $H/32 \times W/32$.
   - `self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)`: replaces `fc7`.
   - `_load_vgg_fc_weights`: reshapes `vgg16.classifier[0].weight` from `[4096, 25088]` to `[4096, 512, 7, 7]`.

3. **`FCN8s.forward` (lines 190–215):**
   - Extract features: `x3` from `stage3` ($H/8, 256$), `x4` from `stage4` ($H/16, 512$), `x5` from `stage5` ($H/32, 512$).
   - `score_fr = self.score_fr(h)`: projects 4096 deep features to $C$ class logits.
   - `upscore2 = self.upscore2(score_fr)`: $2\times$ upsample to stride 16.
   - `fuse_pool4 = upscore2 + self.score_pool4(x4)`: first skip connection (element-wise addition).
   - `upscore_pool4 = self.upscore_pool4(fuse_pool4)`: $2\times$ upsample to stride 8.
   - `fuse_pool3 = upscore_pool4 + self.score_pool3(x3)`: second skip connection with `pool3`.
   - `out = self.upscore8(fuse_pool3)`: final $8\times$ transposed convolution to return to $(H, W)$.

---

## 4. 15 Viva Questions with Direct Answers

#### Q1: Why convert fully connected layers to convolutions? Why not just resize the input to $224 \times 224$?
> **Answer:** If we resize arbitrary images to $224 \times 224$, we distort aspect ratios and lose resolution on high-res images. By converting FC layers to convolutions, the network has no fixed spatial dimensions—it can accept any input size $(H, W)$ and produce a proportional $H \times W$ segmentation map in one forward pass.

#### Q2: Why does FCN use element-wise addition instead of concatenation for skip connections?
> **Answer:** In FCN, each feature map (`pool3`, `pool4`, `conv7`) is first passed through a $1 \times 1$ convolution to project it to $C$ class score channels. Adding them together is equivalent to summing predicted logits from multiple resolutions (like an ensemble across scales). U-Net later used concatenation because it preserves all raw feature channels for subsequent convolutions to learn from.

#### Q3: What is the theoretical receptive field of `conv7`?
> **Answer:** In VGG-16, pooling stages and $3 \times 3$ convolutions expand the receptive field to 212 pixels by `pool5`. The $7 \times 7$ `conv6` adds another $6 \times 32 = 192$ pixels, giving a total receptive field of 404 pixels. This covers typical $352 \times 480$ input images, allowing `conv7` to see global image context.

#### Q4: Why is Mean IoU used instead of Pixel Accuracy?
> **Answer:** Semantic segmentation datasets have severe class imbalance. Road, sky, and buildings account for $>80\%$ of pixels in CamVid, while pedestrians or poles account for $<1\%$. A model predicting "road" everywhere could achieve $80\%$ pixel accuracy while being useless. Mean IoU calculates intersection-over-union per class and averages them, treating rare classes equally.

#### Q5: What does `ignore_index=255` do in CrossEntropyLoss?
> **Answer:** In datasets like CamVid and Pascal VOC, object boundaries are ambiguous, and unlabeled pixels are assigned label 255. `ignore_index=255` masks these pixels out of the loss computation and backpropagation so the model is not penalized on ambiguous boundary pixels.

#### Q6: Why did you initialize transposed convolutions with bilinear weights?
> **Answer:** When initialized randomly, transposed convolutions produce noisy, checkerboard patterns and gradients are chaotic. Bilinear initialization gives the deconvolution a smooth starting point (identical to standard bilinear interpolation). Training then only needs to learn fine boundary corrections rather than learning basic spatial enlargement from scratch.

#### Q7: Why did FCN stop at FCN-8s? Why not FCN-4s or FCN-2s?
> **Answer:** The authors tested FCN-4s in the paper and found almost no mIoU improvement over FCN-8s ($<0.1\%$), while doubling memory consumption. Early layers like `pool1` and `pool2` capture very low-level features (simple edges, color gradients) that lack class semantics, so adding them directly into class scores adds more noise than useful signal.

#### Q8: What causes checkerboard artifacts in transposed convolutions?
> **Answer:** Checkerboard artifacts occur when kernel size is not divisible by stride, creating uneven overlap where some output pixels receive contributions from more weights than others. Using bilinear initialization and kernel sizes that are multiples of stride mitigates this issue.

#### Q9: What is the difference between Transposed Convolution and Dilated (Atrous) Convolution?
> **Answer:** 
> - Transposed convolution increases spatial resolution by expanding features with stride.
> - Dilated convolution keeps spatial resolution constant while expanding the receptive field by inserting holes (dilation rate) between filter weights. Later models like DeepLab use dilated convolutions to avoid downsampling altogether.

#### Q10: How did you handle GPU memory limits during training?
> **Answer:** Because VGG-16's `conv6` outputs 4096 channels, storing intermediate activations takes substantial VRAM. With batch size 8 or 16, it triggers CUDA Out of Memory errors. We set batch size to 4 and used cosine learning rate decay, which ran stably under 6GB of VRAM.

#### Q11: Why is $1 \times 1$ convolution used on `pool4` and `pool3` before skip fusion?
> **Answer:** `pool4` has 512 channels and `pool3` has 256 channels, whereas the upsampled `score_fr` has $C$ channels (11 for CamVid). You cannot add tensors with different channel counts. The $1 \times 1$ convolution projects 512 or 256 channels down to $C$ class scores so they can be added element-wise.

#### Q12: How are the weights of `conv6` initialized from standard VGG-16?
> **Answer:** In PyTorch's pretrained `vgg16`, `classifier[0].weight` is a 2D tensor of shape `(4096, 25088)`. Since $25088 = 512 \times 7 \times 7$, we reshape it to `(4096, 512, 7, 7)` and copy it into `conv6.weight.data`.

#### Q13: In your results, which skip connection gave the largest jump?
> **Answer:** Adding `pool4` in FCN-16s gave the largest improvement (+5.88% mIoU in our run, +3.0% in the paper). Going from 16s to 8s gave an additional +2.71% mIoU, which mostly refined small objects like poles and signs.

#### Q14: What is a key limitation of FCN compared to U-Net?
> **Answer:** FCN uses an asymmetric decoder: it adds class scores directly, whereas U-Net concatenates full high-dimensional feature maps and passes them through multiple decoder convolution blocks. This gives U-Net better boundary reconstruction, especially for medical images.

#### Q15: If you were given more compute, what would you improve?
> **Answer:**
> 1. Replace the VGG-16 backbone with ResNet-50 or ConvNeXt to avoid VGG's heavy 4096-channel memory bottleneck.
> 2. Add modern data augmentation (scale jittering, color jitter, CutMix).
> 3. Replace the deconvolution head with a Feature Pyramid Network (FPN) decoder.

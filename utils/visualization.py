# Deep Learning 2026 (DL-029) - Paper Reproduction
# Roll No: 24CS2038 | Section: IDD
# Utilities for plotting comparisons and training curves


import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from .dataset import decode_segmap, CAMVID_CLASSES


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized PyTorch image tensor [3, H, W] back to uint8 numpy RGB [H, W, 3].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def save_qualitative_comparison(
    image: torch.Tensor,
    gt_mask: np.ndarray,
    pred_32s: np.ndarray,
    pred_16s: np.ndarray,
    pred_8s: np.ndarray,
    save_path: str,
    dataset_name: str = 'camvid',
    title: str = "Semantic Segmentation Progression (FCN-32s vs FCN-16s vs FCN-8s)"
):
    """
    Generates a 5-panel side-by-side comparison mimicking Figure 4 of Long et al. (CVPR 2015):
    [Input Image | Ground Truth | FCN-32s (Coarse) | FCN-16s (Refined) | FCN-8s (Detailed)]
    """
    rgb_img = denormalize_image(image)
    gt_rgb = decode_segmap(gt_mask, dataset_name)
    rgb_32s = decode_segmap(pred_32s, dataset_name)
    rgb_16s = decode_segmap(pred_16s, dataset_name)
    rgb_8s = decode_segmap(pred_8s, dataset_name)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    titles = [
        "Input Image",
        "Ground Truth",
        "FCN-32s (No Skip)",
        "FCN-16s (Pool4 Skip)",
        "FCN-8s (Pool4+Pool3 Skips)"
    ]
    images = [rgb_img, gt_rgb, rgb_32s, rgb_16s, rgb_8s]

    for ax, img, t in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(t, fontsize=13, fontweight='bold')
        ax.axis('off')

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.03)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(history: dict, save_path: str):
    """
    Plots training loss and validation mIoU curves across epochs for all evaluated models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, data in history.items():
        epochs = range(1, len(data['train_loss']) + 1)
        ax1.plot(epochs, data['train_loss'], marker='o', label=f"{model_name}")
        ax2.plot(epochs, data['val_miou'], marker='s', label=f"{model_name}")

    ax1.set_title("Training Loss vs Epochs", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=11)

    ax2.set_title("Validation mIoU (%) vs Epochs", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("mIoU (%)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=11)

    plt.suptitle("FCN Ablation: Training Convergence & Validation Accuracy", fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_iou_comparison(class_names: list, ious_dict: dict, save_path: str):
    """
    Plots a grouped bar chart comparing per-class IoU for FCN-32s, FCN-16s, and FCN-8s.
    """
    x = np.arange(len(class_names))
    n_models = len(ious_dict)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(15, 6))

    for idx, (model_name, ious) in enumerate(ious_dict.items()):
        offset = (idx - n_models / 2 + 0.5) * width
        ax.bar(x + offset, ious, width, label=model_name)

    ax.set_ylabel("Intersection over Union (IoU %)", fontsize=12)
    ax.set_title("Per-Class IoU Comparison: Effect of Skip Connections", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(fontsize=12)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Deep Learning 2026 (DL-029) - Paper Reproduction
# Roll No: 24CS2038 | Section: IDD
# Metrics: Pixel Acc, Mean Acc, mIoU, FW-IoU (Long et al. 2015, Section 5)


import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class Evaluator:
    """
    Maintains a running 2D confusion matrix and computes standard semantic segmentation metrics.
    """
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def _fast_hist(self, label_true: np.ndarray, label_pred: np.ndarray) -> np.ndarray:
        """
        Fast computation of 2D histogram / confusion matrix using 1D binning.
        """
        mask = (label_true >= 0) & (label_true < self.num_classes) & (label_true != self.ignore_index)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def update(self, label_trues: np.ndarray, label_preds: np.ndarray):
        """
        Update the running confusion matrix with a batch of predictions and ground truths.
        Args:
            label_trues: Ground truth masks [B, H, W] or [H, W]
            label_preds: Predicted class masks [B, H, W] or [H, W]
        """
        if isinstance(label_trues, torch.Tensor):
            label_trues = label_trues.detach().cpu().numpy()
        if isinstance(label_preds, torch.Tensor):
            label_preds = label_preds.detach().cpu().numpy()

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def reset(self):
        """Reset the confusion matrix to zeros."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def pixel_accuracy(self) -> float:
        """
        Overall Pixel Accuracy:
        sum_i n_ii / sum_i t_i
        """
        acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-10)
        return float(acc)

    def mean_pixel_accuracy(self) -> float:
        """
        Mean Pixel Accuracy (class-average recall):
        (1 / k) * sum_i (n_ii / t_i)
        """
        acc_cls = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-10)
        return float(np.nanmean(acc_cls))

    def class_ious(self) -> np.ndarray:
        """
        Per-class Intersection over Union (IoU):
        n_ii / (t_i + sum_j n_ji - n_ii)
        """
        diag = np.diag(self.confusion_matrix)
        total_gt = self.confusion_matrix.sum(axis=1)
        total_pred = self.confusion_matrix.sum(axis=0)
        union = total_gt + total_pred - diag
        iou = diag / (union + 1e-10)
        return iou

    def mean_iou(self) -> float:
        """
        Mean Intersection over Union (mIoU):
        (1 / k) * sum_i (IoU_i)
        """
        ious = self.class_ious()
        return float(np.nanmean(ious))

    def frequency_weighted_iou(self) -> float:
        """
        Frequency Weighted IoU:
        (1 / sum_k t_k) * sum_i [t_i * n_ii / (t_i + sum_j n_ji - n_ii)]
        """
        freq = self.confusion_matrix.sum(axis=1) / (self.confusion_matrix.sum() + 1e-10)
        ious = self.class_ious()
        fw_iou = (freq[freq > 0] * ious[freq > 0]).sum()
        return float(fw_iou)

    def get_summary(self, class_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute and return all metrics as a dictionary.
        """
        summary = {
            "Pixel_Accuracy": self.pixel_accuracy() * 100.0,
            "Mean_Pixel_Accuracy": self.mean_pixel_accuracy() * 100.0,
            "mIoU": self.mean_iou() * 100.0,
            "FW_IoU": self.frequency_weighted_iou() * 100.0,
        }
        if class_names is not None and len(class_names) == self.num_classes:
            ious = self.class_ious() * 100.0
            for name, val in zip(class_names, ious):
                summary[f"IoU_{name}"] = float(val)
        return summary

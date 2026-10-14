# Deep Learning 2026 (DL-029) - Paper Reproduction
# Roll No: 24CS2038 | Section: IDD
# Script: run_ablation.py - Compares FCN-32s, 16s, and 8s


import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fcn import build_fcn_model
from utils.dataset import CamVidDataset, CAMVID_CLASSES
from utils.metrics import Evaluator
from utils.visualization import (
    plot_training_curves,
    plot_class_iou_comparison,
    save_qualitative_comparison
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_experiment(model_name: str, num_classes: int, train_loader, val_loader, device, epochs: int, lr: float, init_bilinear: bool = True):
    print(f"\n>>> Running Experiment: {model_name.upper()} (Bilinear Init: {init_bilinear}) <<<")
    model = build_fcn_model(model_name, num_classes=num_classes, pretrained=True, init_bilinear=init_bilinear).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    history = {"train_loss": [], "val_miou": []}
    best_miou = 0.0
    best_summary = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(train_loader))
        history["train_loss"].append(avg_loss)

        # Validation
        model.eval()
        evaluator = Evaluator(num_classes=num_classes, ignore_index=255)
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                evaluator.update(targets.cpu().numpy(), preds.cpu().numpy())

        summary = evaluator.get_summary(class_names=CAMVID_CLASSES)
        history["val_miou"].append(summary["mIoU"])

        if summary["mIoU"] > best_miou:
            best_miou = summary["mIoU"]
            best_summary = summary

        print(f"[{model_name.upper()}] Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f} | Val mIoU: {summary['mIoU']:.2f}% | PixAcc: {summary['Pixel_Accuracy']:.2f}%")

    return model, history, best_summary


def main():
    parser = argparse.ArgumentParser(description="Run FCN Ablation Study (Roll No: 24CS2038)")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per ablation model")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CAMVID_CLASSES)
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare datasets
    train_set = CamVidDataset(root_dir=args.data_dir, split="train", crop_size=(360, 480), augment=True)
    val_set = CamVidDataset(root_dir=args.data_dir, split="val", crop_size=(360, 480), augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    models_to_test = ["fcn32s", "fcn16s", "fcn8s"]
    all_histories = {}
    all_summaries = {}
    trained_models = {}

    for m in models_to_test:
        model, hist, summ = run_single_experiment(
            model_name=m,
            num_classes=num_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            init_bilinear=True
        )
        all_histories[m.upper()] = hist
        all_summaries[m.upper()] = summ
        trained_models[m] = model

    # Plot training loss & mIoU curves
    curves_path = os.path.join(args.output_dir, "ablation_training_curves.png")
    plot_training_curves(all_histories, curves_path)
    print(f"\nSaved training curves to: {curves_path}")

    # Plot Per-Class IoU comparison
    ious_dict = {}
    for m in models_to_test:
        ious_dict[m.upper()] = [all_summaries[m.upper()].get(f"IoU_{c}", 0.0) for c in CAMVID_CLASSES]
    class_bar_path = os.path.join(args.output_dir, "ablation_class_ious.png")
    plot_class_iou_comparison(CAMVID_CLASSES, ious_dict, class_bar_path)
    print(f"Saved class IoU comparison to: {class_bar_path}")

    # Generate Figure 4 Qualitative Progression
    sample_img, sample_mask = val_set[0]
    sample_tensor = sample_img.unsqueeze(0).to(device)

    with torch.no_grad():
        p32 = torch.argmax(trained_models["fcn32s"](sample_tensor), dim=1)[0].cpu().numpy()
        p16 = torch.argmax(trained_models["fcn16s"](sample_tensor), dim=1)[0].cpu().numpy()
        p8 = torch.argmax(trained_models["fcn8s"](sample_tensor), dim=1)[0].cpu().numpy()

    fig4_path = os.path.join(args.output_dir, "qualitative_progression_figure4.png")
    save_qualitative_comparison(
        sample_img,
        sample_mask.numpy(),
        p32,
        p16,
        p8,
        save_path=fig4_path,
        dataset_name='camvid',
        title="Figure 4 Reproduction: Refinement via Skip Connections (FCN-32s -> 16s -> 8s)"
    )
    print(f"Saved qualitative progression to: {fig4_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<12} | {'Pixel Acc (%)':<15} | {'Mean Acc (%)':<15} | {'mIoU (%)':<12} | {'FW-IoU (%)':<12}")
    print("-" * 80)
    for m in ["fcn32s", "fcn16s", "fcn8s"]:
        s = all_summaries[m.upper()]
        print(f"{m.upper():<12} | {s['Pixel_Accuracy']:<15.2f} | {s['Mean_Pixel_Accuracy']:<15.2f} | {s['mIoU']:<12.2f} | {s['FW_IoU']:<12.2f}")
    print("=" * 80)

    # Save summary json
    with open(os.path.join(args.output_dir, "ablation_results.json"), "w") as f:
        json.dump(all_summaries, f, indent=4)


if __name__ == "__main__":
    main()

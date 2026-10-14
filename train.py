# Deep Learning 2026 (DL-029) - Paper Reproduction
# Roll No: 24CS2038 | Section: IDD
# Training script for FCN-32s, FCN-16s, FCN-8s

import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fcn import build_fcn_model
from utils.dataset import CamVidDataset, CAMVID_CLASSES
from utils.metrics import Evaluator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, criterion, optimizer, device, num_classes):
    model.train()
    running_loss = 0.0
    evaluator = Evaluator(num_classes=num_classes, ignore_index=255)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        evaluator.update(targets.cpu().numpy(), preds.cpu().numpy())

    avg_loss = running_loss / max(1, len(loader))
    m_iou = evaluator.mean_iou() * 100.0
    pix_acc = evaluator.pixel_accuracy() * 100.0
    return avg_loss, m_iou, pix_acc


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    evaluator = Evaluator(num_classes=num_classes, ignore_index=255)

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)
        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        evaluator.update(targets.cpu().numpy(), preds.cpu().numpy())

    summary = evaluator.get_summary()
    summary["val_loss"] = running_loss / max(1, len(loader))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fcn8s", choices=["fcn32s", "fcn16s", "fcn8s"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bilinear", action="store_true", help="Ablation: train with random weights instead of bilinear")
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CAMVID_CLASSES)

    print("========================================")
    print(f"DL-029: FCN Implementation (Roll: 24CS2038)")
    print(f"Model: {args.model.upper()} | Device: {device}")
    print(f"Bilinear Init: {not args.no_bilinear} | Batch Size: {args.batch_size}")
    print("========================================")

    # Dataloaders
    train_set = CamVidDataset(root_dir=args.data_dir, split="train", crop_size=(352, 480), augment=True)
    val_set = CamVidDataset(root_dir=args.data_dir, split="val", crop_size=(352, 480), augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = build_fcn_model(
        model_name=args.model,
        num_classes=num_classes,
        pretrained=True,
        init_bilinear=not args.no_bilinear
    ).to(device)

    # Standard pixel-wise cross entropy, 255 = void/unlabeled
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # Paper uses SGD with momentum 0.9 and weight decay 5e-4
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best_miou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_miou": [], "val_pix_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_miou, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        val_res = evaluate_epoch(model, val_loader, criterion, device, num_classes)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_res["val_loss"])
        history["val_miou"].append(val_res["mIoU"])
        history["val_pix_acc"].append(val_res["Pixel_Accuracy"])

        is_best = val_res["mIoU"] > best_miou
        if is_best:
            best_miou = val_res["mIoU"]
            ckpt_path = os.path.join(args.save_dir, f"{args.model}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_miou": best_miou,
                "metrics": val_res,
                "args": vars(args)
            }, ckpt_path)

        dt = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({dt:.1f}s) "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_res['val_loss']:.4f} | "
              f"Val mIoU: {val_res['mIoU']:.2f}% | "
              f"Val PixAcc: {val_res['Pixel_Accuracy']:.2f}% "
              f"{'<- best' if is_best else ''}")

    print(f"\nTraining finished. Best val mIoU: {best_miou:.2f}%")
    log_file = os.path.join(args.save_dir, f"{args.model}_history.json")
    with open(log_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History logged to {log_file}")


if __name__ == "__main__":
    main()

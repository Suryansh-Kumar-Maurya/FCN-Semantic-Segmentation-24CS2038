# Deep Learning 2026 (DL-029) - Paper Reproduction
# Roll No: 24CS2038 | Section: IDD
# Script: evaluate.py - Evaluates checkpoint on validation set

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from models.fcn import build_fcn_model
from utils.dataset import CamVidDataset, CAMVID_CLASSES, VOC_CLASSES, decode_segmap
from utils.metrics import Evaluator
from utils.visualization import denormalize_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fcn8s", choices=["fcn32s", "fcn16s", "fcn8s"])
    parser.add_argument("--dataset", type=str, default="camvid", choices=["camvid", "voc"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = CAMVID_CLASSES if args.dataset == "camvid" else VOC_CLASSES
    num_classes = len(class_names)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_fcn_model(args.model, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_set = CamVidDataset(root_dir=args.data_dir, split="val", crop_size=(352, 480), augment=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    evaluator = Evaluator(num_classes=num_classes, ignore_index=255)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Evaluating {args.model.upper()} on {len(val_set)} validation images...")
    with torch.no_grad():
        for idx, (img_tensor, mask_tensor) in enumerate(val_loader):
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            output = model(img_tensor)
            pred = torch.argmax(output, dim=1)

            evaluator.update(mask_tensor.cpu().numpy(), pred.cpu().numpy())

            if idx < 5:
                # Save first few predictions as images
                rgb_img = denormalize_image(img_tensor[0])
                gt_rgb = decode_segmap(mask_tensor[0].cpu().numpy(), args.dataset)
                pred_rgb = decode_segmap(pred[0].cpu().numpy(), args.dataset)

                Image.fromarray(rgb_img).save(os.path.join(args.output_dir, f"sample_{idx:02d}_img.png"))
                Image.fromarray(gt_rgb).save(os.path.join(args.output_dir, f"sample_{idx:02d}_gt.png"))
                Image.fromarray(pred_rgb).save(os.path.join(args.output_dir, f"sample_{idx:02d}_{args.model}_pred.png"))

    summary = evaluator.get_summary(class_names=class_names)
    print("\n---------------- Evaluation Metrics ----------------")
    print(f"Pixel Accuracy:          {summary['Pixel_Accuracy']:.2f}%")
    print(f"Mean Pixel Accuracy:     {summary['Mean_Pixel_Accuracy']:.2f}%")
    print(f"Mean IoU (mIoU):         {summary['mIoU']:.2f}%")
    print(f"Frequency Weighted IoU:  {summary['FW_IoU']:.2f}%")
    print("----------------------------------------------------")
    print("Per-Class IoU:")
    for name in class_names:
        print(f"  {name:<15}: {summary.get(f'IoU_{name}', 0.0):.2f}%")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()

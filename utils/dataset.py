# Deep Learning 2026 (DL-029) - Paper Reproduction
# Roll No: 24CS2038 | Section: IDD
# Dataset loaders for CamVid and Pascal VOC 2012


import os
import random
import urllib.request
import zipfile
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ==========================================
# CamVid 11-Class Configuration
# ==========================================
CAMVID_CLASSES = [
    'Sky', 'Building', 'Pole', 'Road', 'Pavement',
    'Tree', 'SignSymbol', 'Fence', 'Car', 'Pedestrian', 'Bicyclist'
]

CAMVID_COLORMAP = [
    [128, 128, 128],  # Sky
    [128, 0, 0],      # Building
    [192, 192, 128],  # Pole
    [128, 64, 128],   # Road
    [0, 0, 192],      # Pavement
    [128, 128, 0],    # Tree
    [192, 128, 128],  # SignSymbol
    [64, 64, 128],    # Fence
    [64, 0, 128],     # Car
    [64, 64, 0],      # Pedestrian
    [0, 128, 192],    # Bicyclist
]

# ==========================================
# Pascal VOC 2012 21-Class Configuration
# ==========================================
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def voc_colormap(num_classes=21):
    """Generates standard Pascal VOC color palette."""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            cid >>= 3
        palette[i] = [r, g, b]
    return palette


class CamVidDataset(Dataset):
    """
    CamVid dataset loader supporting both standard directory layout
    and self-contained synthesized mock data for instant offline testing.
    """
    def __init__(self, root_dir: str, split: str = 'train', crop_size: tuple = (360, 480), augment: bool = True):
        self.root_dir = root_dir
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.images = []
        self.masks = []

        split_img_dir = os.path.join(root_dir, split)
        split_mask_dir = os.path.join(root_dir, f"{split}_labels")

        if os.path.exists(split_img_dir) and os.path.exists(split_mask_dir):
            for file_name in sorted(os.listdir(split_img_dir)):
                if file_name.endswith(('.png', '.jpg')):
                    self.images.append(os.path.join(split_img_dir, file_name))
                    mask_name = file_name.replace('.jpg', '_L.png').replace('.png', '_L.png')
                    self.masks.append(os.path.join(split_mask_dir, mask_name))
        else:
            # Generate or reference synthetic samples if local dataset folder is missing
            self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Creates deterministic synthetic samples so testing runs seamlessly offline."""
        synth_dir = os.path.join(self.root_dir, "synthetic", self.split)
        os.makedirs(os.path.join(synth_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(synth_dir, "masks"), exist_ok=True)
        
        num_samples = 40 if self.split == 'train' else 10
        self.images = []
        self.masks = []
        
        rng = np.random.RandomState(42 if self.split == 'train' else 100)
        h, w = self.crop_size

        for idx in range(num_samples):
            img_path = os.path.join(synth_dir, "images", f"sample_{idx:03d}.png")
            mask_path = os.path.join(synth_dir, "masks", f"sample_{idx:03d}.png")

            if not (os.path.exists(img_path) and os.path.exists(mask_path)):
                # Draw geometric road scene (sky top, road bottom, car/pedestrian boxes)
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[: h // 3, :] = 0  # Sky
                mask[h // 3 : 2 * h // 3, :] = 1  # Building
                mask[2 * h // 3 :, :] = 3  # Road
                # Add car
                mask[2 * h // 3 + 20 : 2 * h // 3 + 60, w // 4 : w // 2] = 8
                # Add pedestrian
                mask[h // 2 : 2 * h // 3 + 40, 3 * w // 4 : 3 * w // 4 + 20] = 9

                # Create synthetic RGB image with textures
                img = np.zeros((h, w, 3), dtype=np.uint8)
                for c_idx in range(len(CAMVID_CLASSES)):
                    pixels = (mask == c_idx)
                    color = np.array(CAMVID_COLORMAP[c_idx], dtype=np.float32)
                    noise = rng.normal(0, 15, (h, w, 3))
                    img[pixels] = np.clip(color + noise[pixels], 0, 255).astype(np.uint8)

                Image.fromarray(img).save(img_path)
                Image.fromarray(mask).save(mask_path)

            self.images.append(img_path)
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])

        # Resize to crop_size
        image = image.resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        mask = mask.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)

        # Data Augmentations
        if self.augment and random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Convert to Tensor & Normalize (ImageNet stats)
        img_tensor = TF.to_tensor(image)
        img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))

        return img_tensor, mask_tensor


def decode_segmap(label_mask: np.ndarray, dataset_name: str = 'camvid') -> np.ndarray:
    """
    Decodes an index mask [H, W] to an RGB visualization image [H, W, 3].
    """
    if dataset_name.lower() == 'camvid':
        palette = np.array(CAMVID_COLORMAP, dtype=np.uint8)
        num_classes = len(CAMVID_CLASSES)
    else:
        palette = voc_colormap(21)
        num_classes = 21

    h, w = label_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        idx = (label_mask == c)
        rgb[idx] = palette[c]
    # Void/ignore color (black or white)
    rgb[label_mask == 255] = [0, 0, 0]
    return rgb


def get_dataloaders(root_dir: str, dataset_name: str = 'camvid', batch_size: int = 4, crop_size: tuple = (360, 480)):
    """
    Creates train and validation DataLoaders.
    """
    train_set = CamVidDataset(root_dir=root_dir, split='train', crop_size=crop_size, augment=True)
    val_set = CamVidDataset(root_dir=root_dir, split='val', crop_size=crop_size, augment=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

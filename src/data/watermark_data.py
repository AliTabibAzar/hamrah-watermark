"""Folder-based mask dataset. Layout per source root:

    root/images/<id>.jpg   root/masks/<id>.png   (mask: white = watermark)

Works for CLWD, LOGO-*, PITA-exports, and our synthetic output alike —
anything that follows the layout is trainable with zero code changes.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split


class MaskDataset(Dataset):
    def __init__(self, root: str, size: int = 256, augment: bool = False):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.ids = sorted(f for f in os.listdir(self.img_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name = self.ids[i]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        stem = os.path.splitext(name)[0]
        mask = None
        for ext in (".png", ".jpg"):
            p = os.path.join(self.mask_dir, stem + ext)
            if os.path.exists(p):
                mask = Image.open(p).convert("L")
                break
        if mask is None:
            raise FileNotFoundError(f"No mask for {name} in {self.mask_dir}")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        y = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)[None]
        if self.augment and np.random.rand() < 0.5:
            x, y = x[:, :, ::-1].copy(), y[:, :, ::-1].copy()  # hflip
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return torch.from_numpy((x - mean) / std), torch.from_numpy(y)


def build_loaders(roots: list[str], size: int = 256, batch: int = 16,
                  val_ratio: float = 0.05, workers: int = 2):
    sets = [MaskDataset(r, size=size, augment=True) for r in roots]
    full = sets[0] if len(sets) == 1 else ConcatDataset(sets)
    n_val = max(1, int(len(full) * val_ratio))
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val])
    pin = torch.cuda.is_available()  # pinned memory only helps with CUDA
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True,
                          num_workers=workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=batch, num_workers=workers,
                        pin_memory=pin)
    return train_dl, val_dl

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

try:
    import albumentations as A

    _AUG_STEPS = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15,
                           rotate_limit=15, border_mode=0, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.ImageCompression(quality_range=(60, 100), p=0.3),
    ]
    # GaussNoise changed its std_range semantics across albumentations
    # versions (pixel units vs 0-1). Try new-style first, then old-style,
    # then skip noise entirely — a missing noise step must never kill a run.
    for _noise_kwargs in ({"std_range": (0.05, 0.2)},
                          {"var_limit": (25.0, 400.0)}):
        try:
            _AUG_STEPS.append(A.GaussNoise(p=0.3, **_noise_kwargs))
            break
        except Exception:
            continue
    _AUG = A.Compose(_AUG_STEPS)
except ImportError:
    _AUG = None  # graceful fallback to plain hflip below


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
        img_np = np.asarray(img)
        mask_np = np.asarray(mask)
        if self.augment:
            if _AUG is not None:
                aug = _AUG(image=img_np, mask=mask_np)
                img_np, mask_np = aug["image"], aug["mask"]
            elif np.random.rand() < 0.5:
                img_np = img_np[:, ::-1].copy()
                mask_np = mask_np[:, ::-1].copy()
        x = img_np.astype(np.float32).transpose(2, 0, 1) / 255.0
        y = (mask_np.astype(np.float32) > 127).astype(np.float32)[None]
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

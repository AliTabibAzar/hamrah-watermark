"""U-Net training entry point. Runs on Kaggle T4 (or any CUDA box).

Usage:
    python train_unet.py --data data/clwd data/logo --out models --epochs 40
    python train_unet.py --data data/pita --out models --resume  # continue

Checkpoints (best + every 5 epochs) go to --out. Best = highest val IoU
(loss lies when masks are noisy). Mixed precision on CUDA, fp32 on CPU.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.watermark_data import build_loaders
from src.models.unet import WatermarkUNet, bce_dice_loss


def iou_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * targets).sum().item()
    union = ((pred + targets) > 0).sum().item()
    return inter / max(union, 1)


def atomic_save(state: dict, path: str):
    """Write via temp file + rename: a crash never leaves a corrupt .pt."""
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} cuda={torch.cuda.is_available()}")
    train_dl, val_dl = build_loaders(args.data, size=args.size,
                                     batch=args.batch, workers=args.workers)
    print(f"train_batches={len(train_dl)} val_batches={len(val_dl)}")
    model = WatermarkUNet(pretrained_encoder=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        plateau = False
    else:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
        plateau = True
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    os.makedirs(args.out, exist_ok=True)

    start_epoch, best_iou, bad = 1, 0.0, 0
    best_path = os.path.join(args.out, "watermark-unet.pt")
    if args.resume and os.path.exists(best_path):
        try:
            model.load_state_dict(torch.load(best_path, map_location=device))
            print(f"resumed from {best_path} (weights only, fresh optimizer)")
        except Exception as e:  # noqa: BLE001 — corrupt file must not kill the run
            print(f"resume failed ({e}), starting fresh")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss = bce_dice_loss(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = bce_dice_loss(model(x), y)
                loss.backward()
                opt.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_dl), 1)

        model.eval()
        va_loss, va_iou, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                va_loss += bce_dice_loss(logits, y).item()
                va_iou += iou_score(logits, y)
                n += 1
        va_loss /= max(n, 1)
        va_iou /= max(n, 1)
        if plateau:
            sched.step(va_loss)
        else:
            sched.step()
        print(f"epoch {epoch}/{args.epochs} train={tr_loss:.4f} val={va_loss:.4f} iou={va_iou:.4f}", flush=True)

        if epoch % 5 == 0:
            atomic_save(model.state_dict(), os.path.join(args.out, f"unet_e{epoch}.pt"))
        if va_iou > best_iou + 1e-4:
            best_iou, bad = va_iou, 0
            atomic_save(model.state_dict(), best_path)
            print(f"  -> new best (iou={va_iou:.4f}), saved watermark-unet.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch} (best iou={best_iou:.4f})")
                break
    print(f"done. best val iou={best_iou:.4f} -> {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dataset roots (images/+masks layout)")
    ap.add_argument("--out", default="models")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="continue from --out/watermark-unet.pt if present")
    ap.add_argument("--scheduler", default="plateau", choices=["plateau", "cosine"])
    train(ap.parse_args())

"""
train.py
========
Clean (non-adversarial) baseline training for Stream 2 (LFCC+LCNN)
and Stream 3 (Mel+ResNet18).

Saves best checkpoint to X:\\ (Drive).

Usage:
    python train.py --model lfcc_lcnn
    python train.py --model mel_resnet
    python train.py --model mel_resnet --freeze_backbone
"""

import os, sys, argparse, time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.custom_dataset import build_splits, AudioDataset, MODELS_DIR


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA: {name} ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    print("[Device] CPU")
    return torch.device("cpu")


def load_model(model_name: str, args, device: torch.device) -> nn.Module:
    if model_name == "lfcc_lcnn":
        from models.lfcc_lcnn import LFCCLCNNModel
        return LFCCLCNNModel().to(device)
    elif model_name == "mel_resnet":
        from models.mel_resnet import MelResNetModel
        return MelResNetModel(
            pretrained=True,
            freeze_backbone=getattr(args, "freeze_backbone", False),
        ).to(device)
    else:
        raise ValueError(
            f"Use adversarial_train.py for AASIST. Unknown model: {model_name}"
        )


def _get_logits(model, wav):
    out = model(wav)
    return out[-1] if isinstance(out, (tuple, list)) else out


def train(args):
    device  = get_device()
    use_amp = device.type == "cuda"

    splits = build_splits()
    train_loader = DataLoader(
        AudioDataset(splits["train"]),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=use_amp,
    )
    val_loader = DataLoader(
        AudioDataset(splits["val"]),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=use_amp,
    )

    model     = load_model(args.model, args, device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    loss_fn   = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=use_amp)

    best_val      = 0.0
    patience_left = args.patience
    save_path     = os.path.join(MODELS_DIR, f"{args.model}_best.pt")

    print(f"\nClean Training: {args.model}")
    print(f"  Epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
    print(f"  Train={len(splits['train']):,}  Val={len(splits['val']):,}")
    print(f"  Checkpoint will save to: {save_path}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        t0 = time.time()

        for wav, labels in train_loader:
            wav, labels = wav.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                logits = _get_logits(model, wav)
                loss   = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            correct  += (logits.detach().argmax(1) == labels).sum().item()
            total    += labels.size(0)
            loss_sum += loss.item()

        # Validation
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for wav, labels in val_loader:
                wav, labels = wav.to(device), labels.to(device)
                logits = _get_logits(model, wav)
                vc += (logits.argmax(1) == labels).sum().item()
                vt += labels.size(0)

        tr_acc = correct / total
        vl_acc = vc / vt
        elapsed = time.time() - t0
        print(f"Ep {epoch:3d}/{args.epochs} "
              f"| loss={loss_sum/len(train_loader):.4f} "
              f"| train={tr_acc*100:.2f}% "
              f"| val={vl_acc*100:.2f}% "
              f"| lr={optimizer.param_groups[0]['lr']:.1e} "
              f"| {elapsed:.1f}s")

        scheduler.step(vl_acc)

        if vl_acc > best_val:
            best_val = vl_acc
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          vl_acc,
                "args":             vars(args),
            }, save_path)
            print(f"  [OK] Saved best -> {save_path}")
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop (patience={args.patience})")
                break

    print(f"\nDone. Best val: {best_val*100:.2f}%  |  {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Clean training for LFCC+LCNN / Mel+ResNet18")
    p.add_argument("--model",           required=True, choices=["lfcc_lcnn", "mel_resnet"])
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--patience",        type=int,   default=10)
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze ResNet18 backbone (mel_resnet only)")
    train(p.parse_args())

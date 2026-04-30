"""
adversarial_train.py
====================
PGD-augmented adversarial training for all three streams.
50% of each batch replaced with PGD-attacked audio (eps=0.002, 7 steps).
Mixed precision (AMP) + gradient clipping for RTX 4060 8GB.

Usage:
    # AASIST — fine-tune from existing checkpoint
    python adversarial_train.py --model aasist --checkpoint "X:\\aasist_best_v2.pt"

    # LFCC+LCNN — fine-tune from clean baseline
    python adversarial_train.py --model lfcc_lcnn --checkpoint "X:\\lfcc_lcnn_best.pt"

    # Mel+ResNet18 — fine-tune from clean baseline
    python adversarial_train.py --model mel_resnet --checkpoint "X:\\mel_resnet_best.pt"
"""

import os, sys, argparse, time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.custom_dataset import build_splits, AudioDataset, MODELS_DIR
from attacks.pgd import pgd_attack


def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA: {name} ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    print("[Device] CPU -- adversarial training will be very slow")
    return torch.device("cpu")


def load_model(model_name: str, checkpoint: str, device: torch.device) -> nn.Module:
    if model_name == "lfcc_lcnn":
        from models.lfcc_lcnn import LFCCLCNNModel
        model = LFCCLCNNModel()

    elif model_name == "mel_resnet":
        from models.mel_resnet import MelResNetModel
        model = MelResNetModel(pretrained=True)

    elif model_name == "aasist":
        import importlib.util
        aasist_model_path = os.path.abspath(
            os.path.join("aasist", "models", "AASIST.py")
        )
        if not os.path.isfile(aasist_model_path):
            raise ImportError(
                f"AASIST model file not found: {aasist_model_path}\n"
                "  git clone https://github.com/clovaai/aasist.git"
            )
        aasist_root = os.path.abspath("aasist")
        if aasist_root not in sys.path:
            sys.path.insert(0, aasist_root)
        spec   = importlib.util.spec_from_file_location("AASIST", aasist_model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        AASISTModel = module.Model
        import json
        with open(os.path.join("aasist", "config", "AASIST.conf")) as f:
            cfg = json.load(f)
        model = AASISTModel(cfg["model_config"])

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if checkpoint and os.path.isfile(checkpoint):
        st  = torch.load(checkpoint, map_location=device)
        key = next((k for k in ("model_state_dict", "state_dict") if k in st), None)
        model.load_state_dict(st[key] if key else st, strict=False)
        print(f"[Checkpoint] Loaded: {checkpoint}")
    else:
        print("[Checkpoint] Starting from scratch (no checkpoint found)")

    return model.to(device)


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

    model     = load_model(args.model, args.checkpoint, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )
    loss_fn   = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=use_amp)

    best_val      = 0.0
    patience_left = args.patience
    save_path     = os.path.join(MODELS_DIR, f"{args.model}_adversarial_best.pt")

    print(f"\nAdversarial Training: {args.model}")
    print(f"  Epochs={args.epochs}  bs={args.batch_size}  lr={args.lr}")
    print(f"  PGD: eps={args.pgd_eps}  steps={args.pgd_steps}  adv_frac={args.adv_fraction}")
    print(f"  Train={len(splits['train']):,}  Val={len(splits['val']):,}")
    print(f"  Checkpoint will save to: {save_path}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        t0 = time.time()

        for wav, labels in train_loader:
            wav, labels = wav.to(device), labels.to(device)
            B = wav.size(0)

            # PGD augmentation — replace adv_fraction of each batch
            n_adv = int(B * args.adv_fraction)
            if n_adv > 0:
                idx     = torch.randperm(B, device=device)[:n_adv]
                wav_adv = pgd_attack(
                    model, wav[idx], labels[idx],
                    epsilon=args.pgd_eps,
                    steps=args.pgd_steps,
                    device=device,
                )
                wav = wav.clone()
                wav[idx] = wav_adv

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
            total    += B
            loss_sum += loss.item()

        # Validation (clean)
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
    p = argparse.ArgumentParser(
        description="Adversarial PGD training for AASIST / LFCC+LCNN / Mel+ResNet18"
    )
    p.add_argument("--model",        required=True,
                   choices=["aasist", "lfcc_lcnn", "mel_resnet"])
    p.add_argument("--checkpoint",   default=None,
                   help="Path to .pt checkpoint to fine-tune from")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=16,
                   help="Batch size (16 recommended for RTX 4060 8GB)")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--patience",     type=int,   default=10)
    p.add_argument("--pgd_eps",      type=float, default=0.002,
                   help="PGD epsilon for adversarial augmentation (paper: 0.002)")
    p.add_argument("--pgd_steps",    type=int,   default=7,
                   help="PGD steps per training batch (paper: 7)")
    p.add_argument("--adv_fraction", type=float, default=0.5,
                   help="Fraction of each batch to replace with adversarial examples")
    train(p.parse_args())

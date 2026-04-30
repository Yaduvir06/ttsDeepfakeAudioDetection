"""
eval.py
=======
Full evaluation protocol for any deepfake audio detector.

Runs on the test split (seed=42, 15% of balanced dataset):
  1. Clean accuracy + EER + AUC-ROC
  2. FGSM sweep: eps in {0.0005, 0.001, 0.002, 0.005, 0.01}
  3. PGD sweep (20 steps): same epsilons
  4. MP3 64kbps post-processing attack
  5. MP3 128kbps post-processing attack
  6. Resampling attack: 16kHz -> 8kHz -> 16kHz
  7. Cross-deepfake test: eval only on english_deepfake samples

Results appended to X:\\results\\all_results.csv (append-only, paper source of truth).

Usage:
    python eval.py --model aasist --checkpoint "X:\\aasist_adversarial_v1.pt"
    python eval.py --model lfcc_lcnn --checkpoint "X:\\lfcc_lcnn_best.pt"
    python eval.py --model mel_resnet --checkpoint "X:\\mel_resnet_best.pt"
    python eval.py --model ensemble
"""

import os, sys, argparse, csv
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.custom_dataset import (
    build_splits, AudioDataset, DRIVE_ROOT, DEEPFAKE_DIR, RESULTS_DIR
)
from attacks.fgsm import fgsm_attack
from attacks.pgd  import pgd_attack
from attacks.postprocessing import mp3_attack, resample_attack

EPSILONS = [0.0005, 0.001, 0.002, 0.005, 0.01]


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[Device] CUDA: {name} ({vram:.1f} GB VRAM)")
        return torch.device("cuda")
    print("[Device] CPU (eval will be slow without CUDA)")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_aasist(checkpoint: str, device: torch.device) -> nn.Module:
    """Load AASIST model from clovaai/aasist repo at ./aasist/"""
    import importlib.util

    aasist_model_path = os.path.abspath(
        os.path.join("aasist", "models", "AASIST.py")
    )
    if not os.path.isfile(aasist_model_path):
        raise ImportError(
            f"AASIST model file not found at {aasist_model_path}\n"
            "Clone the repo first:\n"
            "  git clone https://github.com/clovaai/aasist.git"
        )

    # Add aasist/ to sys.path so AASIST.py can import its own dependencies
    aasist_root = os.path.abspath("aasist")
    if aasist_root not in sys.path:
        sys.path.insert(0, aasist_root)

    # Load AASIST.py as a module directly from its file path
    spec   = importlib.util.spec_from_file_location("AASIST", aasist_model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    AASISTModel = module.Model

    import json
    conf_path = os.path.join("aasist", "config", "AASIST.conf")
    with open(conf_path) as f:
        cfg = json.load(f)
    model = AASISTModel(cfg["model_config"])
    _load_checkpoint(model, checkpoint, device)
    return model.to(device)


def _load_checkpoint(model: nn.Module, checkpoint: str, device: torch.device):
    if not checkpoint:
        print("[Checkpoint] No checkpoint — using random weights")
        return
    if not os.path.isfile(checkpoint):
        print(f"[Checkpoint] NOT FOUND: {checkpoint}")
        return
    st = torch.load(checkpoint, map_location=device)
    if isinstance(st, dict):
        key = next((k for k in ("model_state_dict", "state_dict") if k in st), None)
        st = st[key] if key else st
    model.load_state_dict(st, strict=False)
    print(f"[Checkpoint] Loaded: {checkpoint}")


def load_model(model_name: str, checkpoint: str, device: torch.device) -> nn.Module:
    if model_name == "aasist":
        return load_aasist(checkpoint, device)

    elif model_name == "lfcc_lcnn":
        from models.lfcc_lcnn import LFCCLCNNModel
        model = LFCCLCNNModel()
        _load_checkpoint(model, checkpoint, device)
        return model.to(device)

    elif model_name == "mel_resnet":
        from models.mel_resnet import MelResNetModel
        model = MelResNetModel(pretrained=False)
        _load_checkpoint(model, checkpoint, device)
        return model.to(device)

    elif model_name == "ensemble":
        from models.lfcc_lcnn import LFCCLCNNModel
        from models.mel_resnet import MelResNetModel
        from models.ensemble import EnsembleDetector
        s1 = load_aasist(os.path.join(DRIVE_ROOT, "aasist_adversarial_v1.pt"), device)
        s2 = LFCCLCNNModel()
        s3 = MelResNetModel(pretrained=False)
        for m, name in [(s2, "lfcc_lcnn"), (s3, "mel_resnet")]:
            ck = os.path.join(DRIVE_ROOT, f"{name}_adversarial_best.pt")
            _load_checkpoint(m, ck, device)
        model = EnsembleDetector(s1, s2.to(device), s3.to(device))
        return model.to(device)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _get_logits(model: nn.Module, wav: torch.Tensor) -> torch.Tensor:
    """
    Run model forward and extract (B, 2) logits.
    - AASIST returns (last_hidden, logits) tuple -> take [-1]
    - EnsembleDetector returns (final_pred, conflict, stream_preds) -> use _last_avg_logits
    - Others return (B, 2) or (B,) directly
    """
    from models.ensemble import EnsembleDetector
    if isinstance(model, EnsembleDetector):
        # Call forward (stores averaged logits in _last_avg_logits)
        model(wav)
        return model._last_avg_logits   # (B, 2)
    out = model(wav)
    if isinstance(out, (tuple, list)):
        out = out[-1]   # AASIST: (last_hidden, logits)
    return out  # (B, 2) or (B,)


def compute_eer(labels, scores):
    """Compute Equal Error Rate."""
    try:
        fpr, tpr, _ = __import__("sklearn.metrics", fromlist=["roc_curve"]).roc_curve(
            labels, scores, pos_label=1
        )
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return float(eer)
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack_fn=None,
) -> dict:
    """
    Run evaluation with optional adversarial perturbation.

    Returns:
        dict: {accuracy, auc, eer, n_correct, n_total}
    """
    model.eval()
    correct = total = 0
    all_scores, all_labels = [], []

    for wav, labels in loader:
        wav    = wav.to(device)
        labels = labels.to(device)

        if attack_fn is not None:
            wav = attack_fn(wav, labels)

        with torch.no_grad():
            logits = _get_logits(model, wav)
            if logits.dim() == 1:
                # Single-score output: treat as P(bonafide)
                scores = torch.sigmoid(logits)
                preds  = (scores >= 0.5).long()
            else:
                scores = torch.softmax(logits, dim=1)[:, 1]
                preds  = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_scores.extend(scores.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / total if total > 0 else 0.0
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = float("nan")
    eer = compute_eer(all_labels, all_scores)

    return {"accuracy": acc, "auc": auc, "eer": eer,
            "n_correct": correct, "n_total": total}


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def save_result(row: dict):
    # Primary: write to Drive via rclone mount
    # Fallback: write to local project directory if Drive write fails
    LOCAL_RESULTS = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "all_results.csv"
    )
    paths_to_try = [
        os.path.join(RESULTS_DIR, "all_results.csv"),
        LOCAL_RESULTS,
    ]
    saved = False
    for path in paths_to_try:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            write_header = not os.path.isfile(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)
            saved = True
            break
        except OSError as e:
            print(f"  [WARN] Could not write to {path}: {e} — trying fallback...")
    if not saved:
        print(f"  [ERROR] Failed to save result row: {row}")


def log(model_name, attack, epsilon, metrics, timestamp):
    row = {
        "ts":           timestamp,
        "model":        model_name,
        "attack":       attack,
        "epsilon":      epsilon,
        "accuracy_pct": f"{metrics['accuracy']*100:.2f}",
        "auc":          f"{metrics['auc']:.4f}",
        "eer":          f"{metrics['eer']:.4f}",
        "n_total":      metrics["n_total"],
    }
    save_result(row)
    drop = ""
    if attack != "clean":
        drop = f"  (clean - adv = ?)"
    print(f"  {attack:<18s} eps={str(epsilon):<8s} "
          f"acc={metrics['accuracy']*100:.2f}%  "
          f"AUC={metrics['auc']:.4f}  EER={metrics['eer']:.4f}")


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_eval(args):
    device    = get_device()
    model     = load_model(args.model, args.checkpoint, device)
    loss_fn   = nn.CrossEntropyLoss()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build test split
    splits    = build_splits()
    test_samples = splits["test"]

    loader = DataLoader(
        AudioDataset(test_samples),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device.type == "cuda"),
    )

    print(f"\n{'='*65}")
    print(f"Model    : {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test set : {len(test_samples):,} samples")
    print(f"Timestamp: {timestamp}")
    print(f"{'='*65}")

    # 1. Clean
    m = evaluate(model, loader, device)
    clean_acc = m["accuracy"]
    log(args.model, "clean", 0, m, timestamp)

    # For adversarial attacks on the ensemble: use stream1 (AASIST) as the
    # white-box surrogate oracle — it has raw logits suitable for loss.backward().
    # The adversarial wav is then evaluated by the full ensemble.
    from models.ensemble import EnsembleDetector
    attack_oracle = model.stream1 if isinstance(model, EnsembleDetector) else model

    # 2. FGSM sweep
    print("\n-- FGSM sweep --")
    for eps in EPSILONS:
        fn = lambda w, l, e=eps: fgsm_attack(
            attack_oracle, w, l, e, device, loss_fn)
        m = evaluate(model, loader, device, attack_fn=fn)
        log(args.model, "FGSM", eps, m, timestamp)

    # 3. PGD sweep (20 steps)
    print("\n-- PGD-20 sweep --")
    for eps in EPSILONS:
        fn = lambda w, l, e=eps: pgd_attack(
            attack_oracle, w, l, e, 20, device, loss_fn=loss_fn)
        m = evaluate(model, loader, device, attack_fn=fn)
        log(args.model, "PGD-20", eps, m, timestamp)

    # 4-5. MP3 attacks
    print("\n-- Post-processing attacks --")
    for bitrate in ["64k", "128k"]:
        try:
            fn = lambda w, l, b=bitrate: mp3_attack(
                w.cpu(), 16000, b).to(device)
            m = evaluate(model, loader, device, attack_fn=fn)
            log(args.model, f"MP3-{bitrate}", 0, m, timestamp)
        except RuntimeError as e:
            print(f"  MP3-{bitrate} SKIPPED: {e}")

    # 6. Resampling attack
    fn = lambda w, l: resample_attack(w.cpu()).to(device)
    m  = evaluate(model, loader, device, attack_fn=fn)
    log(args.model, "resample-8k", 0, m, timestamp)

    # 7. Cross-deepfake test (english_deepfake samples only)
    print("\n-- Cross-deepfake generalization test --")
    deepfake_samples = [
        (p, l) for p, l in test_samples
        if DEEPFAKE_DIR.lower() in p.lower()
    ]
    if deepfake_samples:
        df_loader = DataLoader(
            AudioDataset(deepfake_samples),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=(device.type == "cuda"),
        )
        m = evaluate(model, df_loader, device)
        log(args.model, "cross-deepfake-clean", 0, m, timestamp)
        print(f"  (deepfake-only test set: {len(deepfake_samples)} samples)")
    else:
        print("  No english_deepfake samples in test split — skipped.")

    csv_path = os.path.join(RESULTS_DIR, "all_results.csv")

    # 8. Ensemble-specific: conflict detection metrics (Table 3)
    from models.ensemble import EnsembleDetector, compute_conflict_metrics
    if isinstance(model, EnsembleDetector):
        print("\n-- Ensemble conflict detection metrics --")
        # pgd_attack is already imported at module level

        def _pgd_fn(wav):
            # conflict metrics helper doesn't pass labels — use dummy ones
            dummy = torch.ones(wav.size(0), dtype=torch.long, device=wav.device)
            return pgd_attack(model.stream1, wav, dummy, 0.002, 20, device, loss_fn=loss_fn)

        clean_loader = loader
        adv_samples  = AudioDataset(test_samples)
        adv_loader   = DataLoader(
            adv_samples, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=(device.type == "cuda"),
        )

        # Compute conflict metrics
        conflict_m = compute_conflict_metrics(model, clean_loader, adv_loader, device)

        # Save to CSV
        row = {
            "ts":           timestamp,
            "model":        "ensemble",
            "attack":       "conflict-detection",
            "epsilon":      0.002,
            "accuracy_pct": "N/A",
            "auc":          f"{conflict_m['conflict_f1']:.4f}",
            "eer":          f"{conflict_m['clean_conflict_rate']:.4f}",
            "n_total":      len(test_samples),
        }
        save_result(row)
        print(f"  Conflict F1={conflict_m['conflict_f1']:.4f}  "
              f"Precision={conflict_m['conflict_precision']:.4f}  "
              f"Recall={conflict_m['conflict_recall']:.4f}")

    print(f"\nAll results saved -> {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Full eval protocol: clean, FGSM, PGD, MP3, resample, cross-deepfake"
    )
    p.add_argument("--model", required=True,
                   choices=["aasist", "lfcc_lcnn", "mel_resnet", "ensemble"])
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt checkpoint. Not needed for --model ensemble.")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Eval batch size (default 16 for RTX 4060 8GB)")
    run_eval(p.parse_args())

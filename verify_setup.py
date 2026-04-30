# -*- coding: utf-8 -*-
"""
verify_setup.py
===============
Run this script at the start of each session to confirm the environment
is ready before training or evaluation.

Checks:
  1. Python version
  2. PyTorch + CUDA (RTX 4060)
  3. Key library imports
  4. rclone Drive mount (X:\)
  5. Protocol files reachable
  6. Existing model checkpoints on Drive

Usage:
    python verify_setup.py
"""

import sys, os

print("=" * 60)
print("Deepfake Audio Detection -- Environment Verification")
print("=" * 60)

# 1. Python
print(f"\n[1] Python {sys.version}")

# 2. PyTorch + CUDA
try:
    import torch
    print(f"\n[2] PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    [OK] CUDA available: {name} ({vram:.1f} GB VRAM)")
    else:
        print("    [!!] CUDA NOT available -- check PyTorch CUDA build")
        print("      Fix: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
except ImportError:
    print("    [!!] PyTorch not installed")

# 3. Key libraries
libs = {
    "torchaudio":   "torchaudio",
    "librosa":      "librosa",
    "sklearn":      "scikit-learn",
    "scipy":        "scipy",
    "soundfile":    "soundfile",
    "einops":       "einops",
    "matplotlib":   "matplotlib",
    "numpy":        "numpy",
}

print("\n[3] Library imports:")
for mod, pkg in libs.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        print(f"    [OK] {pkg} {ver}")
    except ImportError:
        print(f"    [!!] {pkg} -- NOT installed (pip install {pkg})")

try:
    from pydub import AudioSegment
    print("    [OK] pydub")
except ImportError:
    print("    [!!] pydub -- NOT installed (pip install pydub)  [MP3 attack disabled]")

try:
    import torchvision
    print(f"    [OK] torchvision {torchvision.__version__}")
except ImportError:
    print("    [!!] torchvision -- NOT installed (pip install torchvision)  [mel_resnet disabled]")

# 4. rclone Drive mount
DRIVE_ROOT = "X:\\"
print(f"\n[4] Drive mount at {DRIVE_ROOT}:")
if os.path.isdir(DRIVE_ROOT):
    try:
        contents = os.listdir(DRIVE_ROOT)
        print(f"    [OK] Mounted. Contents: {contents[:8]}")
    except Exception as e:
        print(f"    [!!] Mount exists but listdir failed: {e}")
else:
    print("    [!!] Drive NOT mounted.")
    print("      Fix: open a dedicated terminal and run:")
    print("        rclone mount gdrive: X: --vfs-cache-mode full --vfs-cache-max-size 20G")

# 5. Protocol files
PROTOCOL_DIR = os.path.join(DRIVE_ROOT, "aasist_dataset_v2", "protocols")
print(f"\n[5] Protocol files at {PROTOCOL_DIR}:")
for split in ["train", "dev", "eval"]:
    p = os.path.join(PROTOCOL_DIR, f"custom_{split}.txt")
    if os.path.isfile(p):
        with open(p) as f:
            n = sum(1 for _ in f)
        print(f"    [OK] custom_{split}.txt ({n} lines)")
    else:
        print(f"    [!!] custom_{split}.txt -- NOT found")
        print("      Run: python data/protocol_builder.py  (after mounting Drive)")

# 6. Model checkpoints
CHECKPOINTS = {
    "aasist_best_v2.pt":        "Baseline AASIST (Colab)",
    "aasist_adversarial_v1.pt": "Adversarially trained AASIST (Colab)",
    "lfcc_lcnn_best.pt":        "LFCC+LCNN (to be trained locally)",
    "mel_resnet_best.pt":       "Mel+ResNet18 (to be trained locally)",
    "ensemble_best.pt":         "Ensemble (to be built after streams)",
}
print(f"\n[6] Model checkpoints on Drive:")
for fname, desc in CHECKPOINTS.items():
    path = os.path.join(DRIVE_ROOT, fname)
    status = "[OK]" if os.path.isfile(path) else "[--] (missing)"
    print(f"    {status}  {fname}  -- {desc}")

# 7. AASIST repo
print(f"\n[7] AASIST repository:")
if os.path.isdir("aasist"):
    print("    [OK] aasist/ directory exists")
else:
    print("    [!!] aasist/ not found -- clone it:")
    print("      git clone https://github.com/clovaai/aasist.git")

print("\n" + "=" * 60)
print("Verification complete.")
print("=" * 60)

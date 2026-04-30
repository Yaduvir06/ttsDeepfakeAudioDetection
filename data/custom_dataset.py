"""
custom_dataset.py
=================
Glob-based dataset builder — NO protocol .txt files needed.

Drive layout (X:\):
  real   : X:\\english real audio\\LJSpeech-1.1\\wavs\\  (~13,100 .wav)
  TTS    : X:\\english_audio_dataset\\                    (~3,085 files)
  deepfk : X:\\english_deepfake\\                         (~4,382 files)

Split strategy (seed=42, fixed):
  - Fake class = 80% TTS + 20% deepfake
  - Balance: |real| == |fake|  (limited by whichever is smaller)
  - 70% train / 15% val / 15% test — shuffled before split

Labels: 1 = bonafide (real), 0 = spoof (fake)

Usage:
    from data.custom_dataset import build_splits, AudioDataset

    splits = build_splits()   # cached after first call
    train_ds = AudioDataset(splits["train"])
    val_ds   = AudioDataset(splits["val"])
    test_ds  = AudioDataset(splits["test"])
"""

import os
import glob
import random
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Drive path constants — update if mount letter changes
# ---------------------------------------------------------------------------
DRIVE_ROOT   = "X:\\"
REAL_DIR     = os.path.join(DRIVE_ROOT, "english real audio", "LJSpeech-1.1", "wavs")
TTS_FAKE_DIR = os.path.join(DRIVE_ROOT, "english_audio_dataset")
DEEPFAKE_DIR = os.path.join(DRIVE_ROOT, "english_deepfake")
RESULTS_DIR  = os.path.join(DRIVE_ROOT, "results")
MODELS_DIR   = DRIVE_ROOT

AUDIO_EXTS   = ("*.wav", "*.flac", "*.mp3", "*.ogg")

# Cached splits so build_splits() is idempotent within a session
_SPLIT_CACHE: Dict[str, List[Tuple[str, int]]] = {}


def _glob_audio(directory: str) -> List[str]:
    """Recursively collect all audio files under `directory`."""
    files = []
    for ext in AUDIO_EXTS:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        files.extend(glob.glob(os.path.join(directory, ext)))
    # Deduplicate (glob may return duplicates if root matches both patterns)
    return list(dict.fromkeys(files))


def build_splits(
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
    tts_fraction: float = 0.80,   # fraction of fake class that is TTS
    force_rebuild: bool = False,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Build train/val/test splits from Drive audio folders.
    Returns dict: {"train": [(path, label), ...], "val": [...], "test": [...]}

    Results are cached in _SPLIT_CACHE for the session.
    Call with force_rebuild=True to re-scan Drive.
    """
    global _SPLIT_CACHE
    if _SPLIT_CACHE and not force_rebuild:
        return _SPLIT_CACHE

    rng = random.Random(seed)

    print("[Dataset] Scanning Drive audio folders...")
    real_files  = _glob_audio(REAL_DIR)
    tts_files   = _glob_audio(TTS_FAKE_DIR)
    deep_files  = _glob_audio(DEEPFAKE_DIR)

    print(f"  Real files   : {len(real_files):,}")
    print(f"  TTS fakes    : {len(tts_files):,}")
    print(f"  Deepfakes    : {len(deep_files):,}")

    if not real_files:
        raise RuntimeError(f"No real audio found in {REAL_DIR}")
    if not tts_files and not deep_files:
        raise RuntimeError(f"No fake audio found in {TTS_FAKE_DIR} or {DEEPFAKE_DIR}")

    # Shuffle each pool with the fixed seed
    rng.shuffle(real_files)
    rng.shuffle(tts_files)
    rng.shuffle(deep_files)

    # Build balanced fake set: 80% TTS + 20% deepfake
    total_available_fake = len(tts_files) + len(deep_files)
    n_fake = min(len(real_files), total_available_fake)

    n_deepfake = min(int(n_fake * (1.0 - tts_fraction)), len(deep_files))
    n_tts      = min(n_fake - n_deepfake, len(tts_files))
    n_fake     = n_tts + n_deepfake

    # Balance real to match fake count
    n_real = min(n_fake, len(real_files))

    fake_files = tts_files[:n_tts] + deep_files[:n_deepfake]
    real_files = real_files[:n_real]
    rng.shuffle(fake_files)

    print(f"\n  Using {n_real:,} real + {n_fake:,} fake "
          f"({n_tts:,} TTS + {n_deepfake:,} deepfake)")

    # Create labeled pairs and shuffle
    samples: List[Tuple[str, int]] = (
        [(p, 1) for p in real_files] +
        [(p, 0) for p in fake_files]
    )
    rng.shuffle(samples)

    n       = len(samples)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    _SPLIT_CACHE = {
        "train": samples[:n_train],
        "val":   samples[n_train : n_train + n_val],
        "test":  samples[n_train + n_val :],
    }

    for split, data in _SPLIT_CACHE.items():
        n_pos = sum(1 for _, l in data if l == 1)
        n_neg = len(data) - n_pos
        print(f"  {split:5s}: {len(data):,} samples  (real={n_pos:,}, fake={n_neg:,})")

    return _SPLIT_CACHE


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """
    Wraps a list of (path, label) pairs.
    Loads audio from rclone-mounted Drive, resamples to 16kHz mono,
    pads/trims to max_len samples.

    label: 1 = bonafide (real), 0 = spoof (fake)
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        max_len: int  = 64600,
        target_sr: int = 16000,
    ):
        self.samples   = samples
        self.max_len   = max_len
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            wav, sr = torchaudio.load(path, backend="soundfile")
        except Exception:
            try:
                wav, sr = torchaudio.load(path)
            except Exception as e:
                # Return silence on failure — shouldn't happen with rclone
                print(f"  [WARN] Failed to load {path}: {e}")
                return torch.zeros(self.max_len), label

        # Resample to 16kHz
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)

        # Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # (T,)

        # Pad or trim to max_len
        if wav.shape[0] >= self.max_len:
            wav = wav[: self.max_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.max_len - wav.shape[0]))

        return wav, label


def get_loaders(
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
):
    """
    Convenience function: build splits and return DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    splits = build_splits(seed=seed)
    train_loader = DataLoader(
        AudioDataset(splits["train"]),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        AudioDataset(splits["val"]),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        AudioDataset(splits["test"]),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick sanity check — run with: python data/custom_dataset.py
    splits = build_splits()
    ds = AudioDataset(splits["test"][:5])
    for i in range(len(ds)):
        wav, label = ds[i]
        print(f"  [{i}] shape={wav.shape}  label={label}  "
              f"path={splits['test'][i][0][-50:]}")

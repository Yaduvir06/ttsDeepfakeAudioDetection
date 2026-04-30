# Adversarially Robust Deepfake Audio Detection

Research codebase for *"Adversarially Robust Detection of AI-Generated Speech via Multi-Stream Defense Framework"* — targeting ICASSP 2026 / Interspeech 2026.

---

## Project Structure

```
aiVoiceDetectorResearch/
├── data/
│   ├── custom_dataset.py      # Protocol-file-based audio loader (rclone Drive)
│   └── protocol_builder.py    # One-time Colab→Windows path patcher
├── models/
│   ├── lfcc_lcnn.py           # Stream 2: LFCC + LCNN (MFM activations)
│   ├── mel_resnet.py          # Stream 3: Log-mel + ResNet18 (pretrained)
│   └── ensemble.py            # Late-fusion majority vote + conflict detection
├── attacks/
│   ├── fgsm.py                # FGSM with epsilon sweep
│   ├── pgd.py                 # PGD-Linf (20 steps) with random start
│   └── postprocessing.py      # MP3 64k/128k + resampling attacks
├── configs/
│   └── AASIST.conf            # AASIST config (batch_size=16 for RTX 4060)
├── aasist/                    # git clone https://github.com/clovaai/aasist.git
├── train.py                   # Clean training (LFCC+LCNN, Mel+ResNet18)
├── adversarial_train.py       # PGD-augmented adversarial training
├── eval.py                    # Full eval protocol → results/all_results.csv
├── verify_setup.py            # Session startup check (CUDA, Drive, protocols)
└── venvtts/                   # Python 3.12 virtual environment
```

---

## Environment Setup

```powershell
# Activate venv
.\venvtts\Scripts\Activate.ps1

# Verify CUDA (RTX 4060 should show)
python verify_setup.py
```

**PyTorch CUDA build** (already installed):
```
torch 2.5.1+cu121  (CUDA 12.1, Python 3.12, Windows)
```

**Missing packages** (install if needed):
```powershell
pip install pydub seaborn            # MP3 attacks, paper figures
pip install torchvision              # Required for Mel+ResNet18
# ffmpeg must be on PATH for pydub MP3 encoding
```

---

## Session Startup (every session)

```powershell
# Terminal 1 — keep open all session
rclone mount gdrive: G:\GoogleDrive ^
  --vfs-cache-mode full ^
  --vfs-cache-max-size 20G ^
  --vfs-read-ahead 128M ^
  --transfers 8 ^
  --dir-cache-time 72h ^
  --log-level INFO

# Terminal 2 — work here
.\venvtts\Scripts\Activate.ps1
python verify_setup.py
```

---

## Drive Path Constants

| Variable | Path |
|---|---|
| `DRIVE_ROOT` | `G:\GoogleDrive\MyDrive` |
| `REAL_DIR` | `...\english real audio\LJSpeech-1.1\wavs` |
| `TTS_FAKE_DIR` | `...\english_audio_dataset` |
| `DEEPFAKE_DIR` | `...\english_deepfake` |
| `PROTOCOL_DIR` | `...\aasist_dataset_v2\protocols` |
| `RESULTS_DIR` | `...\results` |

---

## Workflow

### Phase 4 — Post-Processing Attacks (current)
```powershell
# Verify aasist_adversarial_v1.pt and fill Table 1
python eval.py --model aasist --checkpoint "G:\GoogleDrive\MyDrive\aasist_adversarial_v1.pt"
```

### Phase 5 — Train New Streams
```powershell
# Stream 2: LFCC + LCNN
python train.py --model lfcc_lcnn
python adversarial_train.py --model lfcc_lcnn --checkpoint "G:\GoogleDrive\MyDrive\lfcc_lcnn_best.pt"

# Stream 3: Mel + ResNet18
python train.py --model mel_resnet
python adversarial_train.py --model mel_resnet --checkpoint "G:\GoogleDrive\MyDrive\mel_resnet_best.pt"
```

### Phase 6 — Ensemble
```powershell
python eval.py --model ensemble
```

---

## Key Hyperparameters

| Param | Value | Notes |
|---|---|---|
| `batch_size` | 16 | RTX 4060 8GB (Colab used 24) |
| `lr` | 1e-4 | Adam, weight_decay=1e-4 |
| `max_len` | 64600 | ~4s @ 16kHz |
| `target_sr` | 16000 | 16kHz mono |
| `pgd_eps` | 0.002 | Adversarial training budget |
| `pgd_steps_train` | 7 | PGD steps per training batch |
| `pgd_steps_eval` | 20 | PGD steps at evaluation |
| `adv_fraction` | 0.5 | 50% of batch is adversarial |
| `patience` | 10 | Early stopping patience |

---

## Results Log

All evaluation results are appended to:
```
G:\GoogleDrive\MyDrive\results\all_results.csv
```

Columns: `ts, model, attack, epsilon, accuracy_pct, auc`

**Never delete this file** — it is the paper's source of truth.

---

## Model Checkpoints

| File | Status | Description |
|---|---|---|
| `aasist_best_v2.pt` | ✅ exists | Baseline AASIST (99.9% clean) |
| `aasist_adversarial_v1.pt` | ✅ exists | PGD-hardened AASIST |
| `lfcc_lcnn_best.pt` | 🔲 to train | Stream 2 baseline |
| `lfcc_lcnn_adversarial_best.pt` | 🔲 to train | Stream 2 hardened |
| `mel_resnet_best.pt` | 🔲 to train | Stream 3 baseline |
| `mel_resnet_adversarial_best.pt` | 🔲 to train | Stream 3 hardened |

---

## Paper Target

- **Title**: *Adversarially Robust Detection of AI-Generated Speech via Multi-Stream Defense Framework*
- **Venue**: ICASSP 2026 (submission ~Sept 2025) or Interspeech 2026
- **Research Q**: Does adversarial PGD training generalise robustness to FGSM, MP3, resampling, and unseen TTS?

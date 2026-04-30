"""
postprocessing.py
=================
Post-processing adversarial attacks:
  - MP3 compression attack (64kbps and 128kbps)
  - Resampling attack (16kHz → 8kHz → 16kHz)

These attacks simulate real-world audio degradation without gradient access,
making them black-box transfer attacks against any trained detector.

Requires: pydub, ffmpeg on PATH.
"""

import io
import torch
import torchaudio
import torchaudio.functional as F
from typing import Tuple

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("[postprocessing] WARNING: pydub not installed — MP3 attack disabled.")


def mp3_attack(
    wav: torch.Tensor,
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> torch.Tensor:
    """
    Encode audio to MP3 at `bitrate` and decode back to PCM.
    Simulates lossy compression artefacts that may fool detectors.

    Args:
        wav         : (B, T) or (T,) float32 tensor in [-1, 1].
        sample_rate : Source sample rate (default 16kHz).
        bitrate     : MP3 bitrate string, e.g. "64k" or "128k".

    Returns:
        wav_out: Same shape as input, float32 in [-1, 1].
    """
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub is not installed. Run: pip install pydub")

    squeeze = (wav.dim() == 1)
    if squeeze:
        wav = wav.unsqueeze(0)  # (1, T)

    batch_size, T = wav.shape
    out_wavs = []

    for i in range(batch_size):
        # Convert to 16-bit PCM bytes
        pcm_int16 = (wav[i].clamp(-1.0, 1.0) * 32767).short().numpy()
        buf_in = io.BytesIO(pcm_int16.tobytes())

        # Build AudioSegment from raw PCM
        seg = AudioSegment.from_raw(
            buf_in,
            sample_width=2,        # 16-bit
            frame_rate=sample_rate,
            channels=1,
        )

        # Encode to MP3
        buf_mp3 = io.BytesIO()
        seg.export(buf_mp3, format="mp3", bitrate=bitrate)
        buf_mp3.seek(0)

        # Decode back to PCM
        seg_decoded = AudioSegment.from_mp3(buf_mp3)
        seg_decoded = seg_decoded.set_channels(1).set_frame_rate(sample_rate)

        samples = torch.frombuffer(
            seg_decoded.raw_data, dtype=torch.int16
        ).float() / 32767.0

        # Match original length
        if samples.shape[0] >= T:
            samples = samples[:T]
        else:
            samples = torch.nn.functional.pad(samples, (0, T - samples.shape[0]))

        out_wavs.append(samples)

    result = torch.stack(out_wavs, dim=0)  # (B, T)
    return result.squeeze(0) if squeeze else result


def resample_attack(
    wav: torch.Tensor,
    orig_sr: int = 16000,
    target_sr: int = 8000,
) -> torch.Tensor:
    """
    Downsample to `target_sr` then upsample back to `orig_sr`.
    This removes high-frequency artefacts that TTS detectors may rely on.

    Args:
        wav       : (B, T) or (T,) float32 tensor.
        orig_sr   : Original sample rate (16000).
        target_sr : Intermediate downsample rate (8000).

    Returns:
        wav_out: Same shape, float32.
    """
    squeeze = (wav.dim() == 1)
    if squeeze:
        wav = wav.unsqueeze(0)  # (1, T)

    # Resample expects (B, C, T) or (C, T); treat batch as channels
    # Process each example individually to avoid shape issues
    out_wavs = []
    for i in range(wav.shape[0]):
        x = wav[i].unsqueeze(0)  # (1, T)
        x_down = F.resample(x, orig_sr, target_sr)
        x_up   = F.resample(x_down, target_sr, orig_sr)
        # Match original length
        T = wav.shape[1]
        if x_up.shape[1] >= T:
            x_up = x_up[:, :T]
        else:
            x_up = torch.nn.functional.pad(x_up, (0, T - x_up.shape[1]))
        out_wavs.append(x_up.squeeze(0))

    result = torch.stack(out_wavs, dim=0)  # (B, T)
    return result.squeeze(0) if squeeze else result


def apply_postprocessing_attacks(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sample_rate: int = 16000,
) -> dict:
    """
    Run all post-processing attacks and return accuracy per attack type.

    Returns:
        dict: {'mp3_64k': acc, 'mp3_128k': acc, 'resample_8k': acc}
    """
    attacks = {
        "mp3_64k":     lambda w: mp3_attack(w.cpu(), sample_rate, "64k").to(device),
        "mp3_128k":    lambda w: mp3_attack(w.cpu(), sample_rate, "128k").to(device),
        "resample_8k": lambda w: resample_attack(w.cpu(), sample_rate, 8000).to(device),
    }

    results = {}

    for attack_name, attack_fn in attacks.items():
        correct = 0
        total = 0

        print(f"  Running {attack_name}...")
        for wav, labels in dataloader:
            wav    = wav.to(device)
            labels = labels.to(device)

            try:
                adv_wav = attack_fn(wav)
            except RuntimeError as e:
                print(f"    ✗ {attack_name} failed: {e}")
                break

            model.eval()
            with torch.no_grad():
                logits = model(adv_wav)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        results[attack_name] = accuracy
        print(f"  {attack_name} → Accuracy: {accuracy*100:.2f}%")

    return results

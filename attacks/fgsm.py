"""
fgsm.py
=======
Fast Gradient Sign Method (FGSM) attack for audio deepfake robustness evaluation.

Usage (within eval loop):
    from attacks.fgsm import fgsm_attack
    adv_wav = fgsm_attack(model, wav, label, epsilon=0.002, device=device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def fgsm_attack(
    model: nn.Module,
    wav: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Single-step FGSM perturbation.

    Args:
        model   : The classifier (must accept raw waveform input).
        wav     : Input waveform batch  (B, T) — already on `device`.
        labels  : Ground-truth labels   (B,)   — already on `device`.
        epsilon : Perturbation magnitude (e.g. 0.002).
        device  : torch.device.
        loss_fn : Loss function. Defaults to CrossEntropyLoss.

    Returns:
        adv_wav : Perturbed waveform (B, T), clamped to [-1, 1].
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.eval()

    wav_adv = wav.detach().clone().requires_grad_(True)

    # Forward pass
    logits = model(wav_adv)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    loss = loss_fn(logits, labels)
    model.zero_grad()
    loss.backward()

    # FGSM perturbation
    with torch.no_grad():
        grad_sign = wav_adv.grad.sign()
        adv_wav = wav_adv + epsilon * grad_sign
        adv_wav = adv_wav.clamp(-1.0, 1.0)

    return adv_wav.detach()


def fgsm_sweep(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilons: list,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> dict:
    """
    Run FGSM at multiple epsilon values and return accuracy per epsilon.

    Returns:
        dict: {epsilon (float): accuracy (float 0-1)}
    """
    results = {}

    for eps in epsilons:
        correct = 0
        total = 0

        for wav, labels in dataloader:
            wav    = wav.to(device)
            labels = labels.to(device)

            adv_wav = fgsm_attack(model, wav, labels, eps, device, loss_fn)

            model.eval()
            with torch.no_grad():
                logits = model(adv_wav)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        results[eps] = accuracy
        print(f"  FGSM ε={eps:.4f} → Accuracy: {accuracy*100:.2f}%")

    return results

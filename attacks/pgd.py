"""
pgd.py
======
Projected Gradient Descent (PGD) attack for audio deepfake robustness evaluation.
PGD is a stronger, iterated version of FGSM — it is the primary robustness benchmark
used in this research (see deepfake_audio_worflow.md §7).

Usage:
    from attacks.pgd import pgd_attack, pgd_sweep
    adv_wav = pgd_attack(model, wav, labels, epsilon=0.002, steps=20, device=device)
"""

import torch
import torch.nn as nn
from typing import Optional


def pgd_attack(
    model: nn.Module,
    wav: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    steps: int,
    device: torch.device,
    step_size: Optional[float] = None,
    loss_fn: Optional[nn.Module] = None,
    random_start: bool = True,
) -> torch.Tensor:
    """
    PGD-Linf adversarial perturbation.

    Args:
        model       : Classifier accepting raw waveform (B, T).
        wav         : Input batch (B, T) on `device`.
        labels      : Ground-truth (B,) on `device`.
        epsilon     : L-inf budget.
        steps       : Number of PGD iterations (paper uses 20).
        device      : torch.device.
        step_size   : Per-step alpha. Defaults to epsilon/steps * 2.5.
        loss_fn     : Defaults to CrossEntropyLoss.
        random_start: Start from random point inside epsilon ball (recommended).

    Returns:
        adv_wav: Perturbed waveform (B, T), clamped to [-1, 1] ∩ [wav±ε].
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if step_size is None:
        step_size = epsilon * 2.5 / steps

    model.eval()
    wav_orig = wav.detach().clone()

    # Optional random start within the epsilon ball
    if random_start:
        delta = torch.empty_like(wav_orig).uniform_(-epsilon, epsilon)
        wav_adv = (wav_orig + delta).clamp(-1.0, 1.0).detach()
    else:
        wav_adv = wav_orig.clone()

    for _ in range(steps):
        wav_adv.requires_grad_(True)

        logits = model(wav_adv)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        loss = loss_fn(logits, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_sign = wav_adv.grad.sign()
            wav_adv = wav_adv + step_size * grad_sign
            # Project back into epsilon ball around original
            delta = torch.clamp(wav_adv - wav_orig, -epsilon, epsilon)
            wav_adv = (wav_orig + delta).clamp(-1.0, 1.0)

    return wav_adv.detach()


def pgd_sweep(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    epsilons: list,
    device: torch.device,
    steps: int = 20,
    loss_fn: Optional[nn.Module] = None,
) -> dict:
    """
    Run PGD at multiple epsilon values and return accuracy per epsilon.

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

            adv_wav = pgd_attack(model, wav, labels, eps, steps, device, loss_fn=loss_fn)

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
        print(f"  PGD-{steps} ε={eps:.4f} → Accuracy: {accuracy*100:.2f}%")

    return results

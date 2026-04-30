"""
ensemble.py
===========
Phase 6: Late-fusion ensemble with majority voting + conflict detection.

Combines three streams:
  Stream 1: AASIST (graph-based, loaded from checkpoint)
  Stream 2: LFCC + LCNN
  Stream 3: Mel-spectrogram + ResNet18

Conflict Detection:
  - If streams disagree (vote is split e.g. 2v1), the sample is flagged
    as "potentially adversarial" for downstream investigation.
  - Conflict rate on clean vs adversarial inputs is itself a metric
    (high conflict rate → model is detecting something unusual).

Output per sample:
  - final_pred: 0 (spoof) or 1 (bonafide)
  - conflict_flag: True if any stream disagreed
  - vote_distribution: {stream: pred}
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np


class EnsembleDetector(nn.Module):
    """
    Late-fusion majority-vote ensemble of up to 3 deepfake detectors.

    Streams are registered as submodules so their parameters are tracked
    by the PyTorch optimizer if fine-tuning is desired.

    Conflict Detection:
        Conflict is flagged when NOT all streams agree on the same prediction.
        For 3 streams: conflict = any vote split (2-1 or hypothetically 1-1-1).
    """

    def __init__(
        self,
        stream1: nn.Module,   # AASIST
        stream2: nn.Module,   # LFCC+LCNN
        stream3: nn.Module,   # Mel+ResNet18
        stream_names: List[str] = None,
    ):
        super().__init__()
        self.stream1 = stream1
        self.stream2 = stream2
        self.stream3 = stream3
        self.stream_names = stream_names or ["aasist", "lfcc_lcnn", "mel_resnet"]

    def _get_stream_logits(self, stream: nn.Module, wav: torch.Tensor) -> torch.Tensor:
        """Run a stream and extract (B, 2) logits, handling AASIST's (hidden, logits) tuple."""
        out = stream(wav)
        if isinstance(out, (tuple, list)):
            out = out[-1]   # AASIST returns (last_hidden, logits) — take last
        return out  # (B, 2)

    @torch.no_grad()
    def forward(
        self,
        wav: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            wav: (B, T) raw waveform on the appropriate device.

        Returns:
            final_pred  : (B,) — majority vote prediction (0=spoof, 1=bonafide)
            conflict    : (B,) bool — True if any stream disagreed
            stream_preds: dict of {stream_name: (B,) predictions}

        NOTE: eval.py calls _get_logits(model, wav) which checks for tuple/list
        and takes [-1]. For the ensemble we return a tuple (final_pred, conflict,
        stream_preds) so eval.py will take final_pred as logits. To keep eval.py
        compatible, we also expose a logits-style path via __call__ returning
        averaged soft logits when used by eval._get_logits.
        """
        streams = [self.stream1, self.stream2, self.stream3]
        preds_list  = []
        logits_list = []

        for stream in streams:
            stream.eval()
            logits = self._get_stream_logits(stream, wav)   # (B, 2)
            logits_list.append(logits)
            preds_list.append(logits.argmax(dim=1))          # (B,)

        # Stack votes → (3, B)
        votes = torch.stack(preds_list, dim=0)

        # Majority vote
        vote_sum   = votes.sum(dim=0).float()          # (B,) in [0, 3]
        final_pred = (vote_sum >= 1.5).long()          # majority of 3

        # Conflict: not all streams agree
        conflict = ~((votes == votes[0]).all(dim=0))   # (B,) bool

        stream_preds = {
            name: preds_list[i]
            for i, name in enumerate(self.stream_names)
        }

        # Store averaged soft logits so eval.py's _get_logits can use them
        self._last_avg_logits = torch.stack(logits_list, dim=0).mean(dim=0)  # (B,2)

        return final_pred, conflict, stream_preds

    @torch.no_grad()
    def get_soft_scores(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Average softmax probability across streams.
        Returns (B, 2) averaged probability distribution.
        Useful for EER computation.
        """
        streams = [self.stream1, self.stream2, self.stream3]
        probs_list = []

        for stream in streams:
            stream.eval()
            logits = self._get_stream_logits(stream, wav)   # (B, 2)
            probs  = torch.softmax(logits, dim=1)            # (B, 2)
            probs_list.append(probs)

        avg_probs = torch.stack(probs_list, dim=0).mean(dim=0)  # (B, 2)
        return avg_probs


# ---------------------------------------------------------------------------
# Ensemble Evaluation
# ---------------------------------------------------------------------------

def evaluate_ensemble(
    ensemble: EnsembleDetector,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    attack_fn=None,
    attack_name: str = "clean",
) -> dict:
    """
    Run full evaluation of the ensemble on a dataloader.

    Args:
        ensemble    : EnsembleDetector instance.
        dataloader  : Eval data loader.
        device      : CUDA or CPU.
        attack_fn   : Optional callable (wav, labels) → adv_wav.
        attack_name : String label for logging.

    Returns:
        dict with keys: accuracy, conflict_rate, per_stream_accuracy
    """
    all_preds   = []
    all_labels  = []
    all_conflicts = []
    stream_correct = {name: 0 for name in ensemble.stream_names}
    total = 0

    for wav, labels in dataloader:
        wav    = wav.to(device)
        labels = labels.to(device)

        if attack_fn is not None:
            wav = attack_fn(wav)

        final_pred, conflict, stream_preds = ensemble(wav)

        all_preds.append(final_pred.cpu())
        all_labels.append(labels.cpu())
        all_conflicts.append(conflict.cpu())

        for name in ensemble.stream_names:
            stream_correct[name] += (stream_preds[name] == labels).sum().item()
        total += labels.size(0)

    all_preds   = torch.cat(all_preds)
    all_labels  = torch.cat(all_labels)
    all_conflicts = torch.cat(all_conflicts)

    accuracy     = (all_preds == all_labels).float().mean().item()
    conflict_rate = all_conflicts.float().mean().item()
    per_stream   = {n: stream_correct[n] / total for n in ensemble.stream_names}

    print(f"\n[Ensemble] {attack_name}")
    print(f"  Accuracy       : {accuracy*100:.2f}%")
    print(f"  Conflict rate  : {conflict_rate*100:.2f}%")
    for name, acc in per_stream.items():
        print(f"  {name:15s} : {acc*100:.2f}%")

    return {
        "attack": attack_name,
        "accuracy": accuracy,
        "conflict_rate": conflict_rate,
        "per_stream_accuracy": per_stream,
    }


# ---------------------------------------------------------------------------
# Conflict Detection Metrics
# ---------------------------------------------------------------------------

def compute_conflict_metrics(
    ensemble: EnsembleDetector,
    clean_loader: torch.utils.data.DataLoader,
    adv_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """
    Compare conflict rates on clean vs adversarial inputs.
    Higher conflict on adversarial suggests the ensemble can flag attacks.

    Returns precision/recall of conflict as an adversarial detector.
    """
    def get_conflict_flags(loader, is_adversarial=False):
        flags = []
        for wav, labels in loader:
            wav = wav.to(device)
            _, conflict, _ = ensemble(wav)
            flags.extend(conflict.cpu().tolist())
        return flags

    clean_flags = get_conflict_flags(clean_loader, is_adversarial=False)
    adv_flags   = get_conflict_flags(adv_loader,   is_adversarial=True)

    # Treat conflict as a binary adversarial detector:
    # True Positive: adversarial sample flagged as conflict
    # False Positive: clean sample flagged as conflict
    tp = sum(adv_flags)
    fp = sum(clean_flags)
    fn = len(adv_flags) - tp
    tn = len(clean_flags) - fp

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n[Conflict Detection Metrics]")
    print(f"  Clean conflict rate : {sum(clean_flags)/len(clean_flags)*100:.2f}%")
    print(f"  Adv conflict rate   : {sum(adv_flags)/len(adv_flags)*100:.2f}%")
    print(f"  Precision           : {precision:.4f}")
    print(f"  Recall              : {recall:.4f}")
    print(f"  F1                  : {f1:.4f}")

    return {
        "clean_conflict_rate":     sum(clean_flags) / len(clean_flags),
        "adv_conflict_rate":       sum(adv_flags) / len(adv_flags),
        "conflict_precision":      precision,
        "conflict_recall":         recall,
        "conflict_f1":             f1,
    }

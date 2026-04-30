"""
lfcc_lcnn.py
============
Stream 2: LFCC feature extractor + LCNN classifier.

LFCC (Linear Frequency Cepstral Coefficients) capture different spectral cues
than AASIST's graph-based features, providing complementary information for
the ensemble.

LCNN (Light CNN) is a standard anti-spoofing architecture using Max-Feature-Map
(MFM) activations instead of ReLU — shown to be effective in ASVspoof challenges.

Reference:
  Wu et al. (2020), "Light CNN for deep face representation with noisy labels"
  — adapted here for spectral anti-spoofing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ---------------------------------------------------------------------------
# LFCC Feature Extractor
# ---------------------------------------------------------------------------

class LFCCFeatureExtractor(nn.Module):
    """
    Extract LFCC features from raw waveform using torchaudio's built-in LFCC.

    Output: (B, n_lfcc, T_frames)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 70,
        n_lfcc: int = 60,
        f_min: float = 0.0,
        f_max: float = None,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        log_lf: bool = False,
    ):
        super().__init__()
        f_max = f_max or sample_rate / 2

        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            f_min=f_min,
            f_max=f_max,
            speckwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "window_fn": torch.hann_window,
            },
            log_lf=log_lf,
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) raw waveform.
        Returns:
            feats: (B, n_lfcc, T_frames)
        """
        return self.lfcc(wav)  # (B, n_lfcc, T_frames)


# ---------------------------------------------------------------------------
# Max-Feature-Map (MFM) activation
# ---------------------------------------------------------------------------

class MaxFeatureMap2D(nn.Module):
    """
    MFM activation: splits channel dim in half and takes element-wise max.
    Reduces channels by factor 2. Acts as a learned gating mechanism.
    """

    def __init__(self, max_dim: int = 1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        assert shape[self.max_dim] % 2 == 0, (
            f"MaxFeatureMap: dim {self.max_dim} must be even, got {shape[self.max_dim]}"
        )
        shape[self.max_dim] //= 2
        x = x.view(*shape[:self.max_dim], 2, *shape[self.max_dim:])
        return x.max(dim=self.max_dim)[0]


# ---------------------------------------------------------------------------
# LCNN Classifier
# ---------------------------------------------------------------------------

class LCNN(nn.Module):
    """
    Light CNN anti-spoofing classifier.
    Input: LFCC feature maps (B, 1, n_lfcc, T_frames).
    Output: logits (B, 2).
    """

    def __init__(self, n_lfcc: int = 60, n_frames: int = 400, n_classes: int = 2):
        super().__init__()
        self.n_lfcc   = n_lfcc
        self.n_frames = n_frames

        # Block 1: Conv → MFM → Pool
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.mfm1  = MaxFeatureMap2D()           # 64 → 32 ch
        self.pool1 = nn.MaxPool2d(2, 2)          # /2 spatial

        # Block 2
        self.conv2a = nn.Conv2d(32, 64, kernel_size=1)
        self.mfm2a  = MaxFeatureMap2D()          # 64 → 32 ch
        self.conv2  = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.mfm2   = MaxFeatureMap2D()          # 128 → 64 ch
        self.pool2  = nn.MaxPool2d(2, 2)

        # Block 3
        self.conv3a = nn.Conv2d(64, 128, kernel_size=1)
        self.mfm3a  = MaxFeatureMap2D()          # 128 → 64 ch
        self.conv3  = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.mfm3   = MaxFeatureMap2D()          # 128 → 64 ch
        self.pool3  = nn.MaxPool2d(2, 2)

        # Block 4
        self.conv4a = nn.Conv2d(64, 64, kernel_size=1)
        self.mfm4a  = MaxFeatureMap2D()          # 64 → 32 ch
        self.conv4  = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.mfm4   = MaxFeatureMap2D()          # 64 → 32 ch
        self.pool4  = nn.MaxPool2d(2, 2)

        # Block 5 (no pool)
        self.conv5a = nn.Conv2d(32, 32, kernel_size=1)
        self.mfm5a  = MaxFeatureMap2D()          # 32 → 16 ch
        self.conv5  = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.mfm5   = MaxFeatureMap2D()          # 32 → 16 ch

        # Global average pool + FC
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(16, 32)
        self.mfm6 = MaxFeatureMap2D()            # 32 → 16
        self.drop = nn.Dropout(p=0.75)
        self.fc2  = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_lfcc, T_frames) — LFCC feature map.
        Returns:
            logits: (B, n_classes)
        """
        x = self.pool1(self.mfm1(self.conv1(x)))

        x = self.mfm2a(self.conv2a(x))
        x = self.pool2(self.mfm2(self.conv2(x)))

        x = self.mfm3a(self.conv3a(x))
        x = self.pool3(self.mfm3(self.conv3(x)))

        x = self.mfm4a(self.conv4a(x))
        x = self.pool4(self.mfm4(self.conv4(x)))

        x = self.mfm5a(self.conv5a(x))
        x = self.mfm5(self.conv5(x))

        x = self.gap(x).flatten(1)      # (B, 16)
        x = self.mfm6(self.fc1(x))     # (B, 16)
        x = self.drop(x)
        x = self.fc2(x)                 # (B, n_classes)
        return x


# ---------------------------------------------------------------------------
# Combined LFCC+LCNN Model
# ---------------------------------------------------------------------------

class LFCCLCNNModel(nn.Module):
    """
    End-to-end LFCC + LCNN model.
    Input: raw waveform (B, T).
    Output: logits (B, 2).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_lfcc: int = 60,
        n_frames: int = 400,
        n_classes: int = 2,
    ):
        super().__init__()
        self.feature_extractor = LFCCFeatureExtractor(
            sample_rate=sample_rate, n_lfcc=n_lfcc
        )
        self.classifier = LCNN(n_lfcc=n_lfcc, n_frames=n_frames, n_classes=n_classes)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(wav)          # (B, n_lfcc, T_frames)
        feats = feats.unsqueeze(1)                   # (B, 1, n_lfcc, T_frames)
        return self.classifier(feats)                # (B, n_classes)

    def get_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        """Return the 16-dim embedding before the final FC (for ensemble fusion)."""
        feats = self.feature_extractor(wav).unsqueeze(1)
        x = self.classifier.pool1(self.classifier.mfm1(self.classifier.conv1(feats)))
        x = self.classifier.mfm2a(self.classifier.conv2a(x))
        x = self.classifier.pool2(self.classifier.mfm2(self.classifier.conv2(x)))
        x = self.classifier.mfm3a(self.classifier.conv3a(x))
        x = self.classifier.pool3(self.classifier.mfm3(self.classifier.conv3(x)))
        x = self.classifier.mfm4a(self.classifier.conv4a(x))
        x = self.classifier.pool4(self.classifier.mfm4(self.classifier.conv4(x)))
        x = self.classifier.mfm5a(self.classifier.conv5a(x))
        x = self.classifier.mfm5(self.classifier.conv5(x))
        x = self.classifier.gap(x).flatten(1)
        x = self.classifier.mfm6(self.classifier.fc1(x))
        return x  # (B, 16)

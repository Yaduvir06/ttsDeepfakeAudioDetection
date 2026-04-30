"""
mel_resnet.py
=============
Stream 3: Mel-spectrogram + ResNet18 classifier.

Uses a pretrained ResNet18 backbone (ImageNet weights) adapted for
single-channel log-mel spectrograms. This transfer-learning approach
leverages visual pattern recognition on 2D spectral representations,
which captures different artifact patterns than AASIST or LFCC+LCNN.

Input: raw waveform (B, T)
Output: logits (B, 2)
"""

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


# ---------------------------------------------------------------------------
# Mel-spectrogram Feature Extractor
# ---------------------------------------------------------------------------

class MelSpectrogramExtractor(nn.Module):
    """
    Converts raw waveform to a log-mel spectrogram suitable for ResNet18.

    Output shape: (B, 1, n_mels, T_frames)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 128,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        top_db: float = 80.0,
    ):
        super().__init__()
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            window_fn=torch.hann_window,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=top_db)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav: (B, T) float32 waveform.
        Returns:
            mel_db: (B, 1, n_mels, T_frames) log-mel spectrogram.
        """
        mel = self.mel_spec(wav)           # (B, n_mels, T_frames)
        mel_db = self.amplitude_to_db(mel) # log scale
        return mel_db.unsqueeze(1)         # (B, 1, n_mels, T_frames)


# ---------------------------------------------------------------------------
# ResNet18-based Classifier
# ---------------------------------------------------------------------------

class MelResNet18(nn.Module):
    """
    ResNet18 adapted for single-channel log-mel spectrogram classification.

    Architecture:
    - Adapts the first Conv2d layer to accept 1 channel (instead of 3 RGB).
    - Replaces the final FC layer for binary classification.
    - Uses ImageNet pretrained weights (excluding first conv and final fc).
    """

    def __init__(
        self,
        n_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Load ResNet18
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        except ImportError:
            raise ImportError("torchvision is required. Run: pip install torchvision")

        # Adapt first conv: 3-channel → 1-channel (average RGB weights)
        orig_conv = backbone.conv1
        new_conv = nn.Conv2d(
            1, 64,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        if pretrained:
            # Average the 3 RGB weight channels → 1 channel
            new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # Replace final FC layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, n_classes),
        )

        self.backbone = backbone

        if freeze_backbone:
            # Freeze all layers except the final FC
            for name, param in self.backbone.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, T_frames) log-mel spectrogram.
        Returns:
            logits: (B, n_classes)
        """
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-dim embedding before final FC (for ensemble fusion)."""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer1(x)
        x = b.layer2(x)
        x = b.layer3(x)
        x = b.layer4(x)
        x = b.avgpool(x)
        return torch.flatten(x, 1)  # (B, 512)


# ---------------------------------------------------------------------------
# Combined Mel + ResNet18 End-to-End Model
# ---------------------------------------------------------------------------

class MelResNetModel(nn.Module):
    """
    End-to-end: raw waveform → log-mel → ResNet18 → logits.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.feature_extractor = MelSpectrogramExtractor(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )
        self.classifier = MelResNet18(
            n_classes=n_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        # Resize transform to make spectrograms compatible with ResNet input
        self.resize = nn.AdaptiveAvgPool2d((224, 224))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.feature_extractor(wav)  # (B, 1, n_mels, T_frames)
        mel = self.resize(mel)             # (B, 1, 224, 224)
        return self.classifier(mel)        # (B, n_classes)

    def get_embedding(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.feature_extractor(wav)
        mel = self.resize(mel)
        return self.classifier.get_embedding(mel)  # (B, 512)

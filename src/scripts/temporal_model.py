import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MyTemporalModel(nn.Module):
    """
    Temporal fall detector: ResNet50 backbone + LSTM + MLP head.
    Expects input of shape (B, seq_len, 3, H, W), and outputs (B, num_classes).
    """
    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 256,
        pretrained_backbone: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        # 1) ResNet50 backbone (no FC) for feature extraction
        weights = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        resnet = resnet50(weights=weights)
        # strip off the last fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # 2) LSTM over the 2048-dim features
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        # 3) Classifier head: two-layer MLP with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, 3, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        B, S, C, H, W = x.shape
        # collapse batch+sequence for backbone
        x = x.view(B * S, C, H, W)
        # [B*S, 2048, 1, 1]
        feats = self.backbone(x)
        # [B, S, 2048]
        feats = feats.view(B, S, -1)
        # LSTM -> [B, S, hidden_size]
        lstm_out, _ = self.lstm(feats)
        # take last timestep -> [B, hidden_size]
        final_state = lstm_out[:, -1, :]
        # classifier -> [B, num_classes]
        return self.classifier(final_state)

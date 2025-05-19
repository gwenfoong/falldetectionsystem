import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MyBaselineModel(nn.Module):
    """
    Frame-level fall detector based on ResNet50.
    Loads ImageNet-pretrained weights by default and replaces the final FC.
    """
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        # Choose weights enum if pretrained, else None
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        # Load the ResNet50 backbone
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features  # usually 2048
        # Replace the final fully-connected layer
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, 3, H, W)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        return self.backbone(x)

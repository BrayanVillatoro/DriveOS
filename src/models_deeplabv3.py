import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class RacingLineDeepLabV3(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        self.segmentation = deeplabv3_resnet50(pretrained=pretrained)
        # Replace the classifier head for your number of classes
        self.segmentation.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, 3, H, W]
        out = self.segmentation(x)['out']  # [B, num_classes, H, W]
        return out

# Example usage:
# model = RacingLineDeepLabV3(num_classes=3)
# output = model(torch.randn(1, 3, 320, 320))
# output shape: [1, 3, 320, 320]

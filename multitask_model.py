import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.segmentation import deeplabv3_resnet50


class FederatedMultiTaskModel(nn.Module):
    def __init__(self, num_seg_classes=4, num_disease_classes=5):
        super().__init__()

        self.seg_model = deeplabv3_resnet50(weights=None, weights_backbone=None)
        self.seg_model.classifier[4] = nn.Conv2d(256, num_seg_classes, kernel_size=1)

        backbone = resnet50(weights=None)
        self.classifier_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier_head = nn.Linear(2048, num_disease_classes)

    def forward(self, x):
        seg_logits = self.seg_model(x)["out"]

        feat = self.classifier_backbone(x)
        feat = torch.flatten(feat, 1)
        disease_logits = self.classifier_head(feat)

        return {
            "segmentation": seg_logits,
            "disease": disease_logits
        }
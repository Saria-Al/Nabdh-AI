import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


def build_deeplab(num_seg_classes=4):
    model = deeplabv3_resnet50(weights=None, num_classes=num_seg_classes)

    old = model.backbone.conv1
    new = nn.Conv2d(
        1,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=False
    )

    with torch.no_grad():
        new.weight[:] = old.weight.mean(dim=1, keepdim=True)

    model.backbone.conv1 = new
    return model
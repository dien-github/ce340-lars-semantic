import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.segmentation.deeplabv3 import DeepLabV3 # Can remove if not used
# from torchvision.models.segmentation.lraspp import LRASPP # Can remove if not used
# Remove the problematic import
# from torchvision.models import MobileNetV3Large_Weights # <--- REMOVE THIS LINE


class FSCNN_MobileNetV3(nn.Module):
    """
    FSCNN-like architecture with MobileNetV3 as the backbone for semantic segmentation.
    This aims to be a simplified FCN/Deeplab-like structure using MobileNetV3 features.
    """

    def __init__(self, num_classes, pretrained_backbone=True):
        super(FSCNN_MobileNetV3, self).__init__()

        # Load MobileNetV3Large using the 'pretrained' boolean argument
        # This is compatible with older torchvision versions.
        backbone = models.mobilenet_v3_large(
            pretrained=pretrained_backbone, dilated=True
        )  # <--- MODIFIED LINE

        return_layers = {"features.16": "out", "features.12": "low_level"}

        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

        inplanes = 960
        low_level_inplanes = 160

        self.classifier = FCNHead(inplanes, num_classes)

        self.conv_low_level = nn.Conv2d(low_level_inplanes, 48, kernel_size=1)
        self.concat_conv = nn.Conv2d(
            num_classes + 48, num_classes, kernel_size=3, padding=1
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        x = features["out"]
        x = self.classifier(x)

        low_level_features = features["low_level"]
        low_level_features = self.conv_low_level(low_level_features)

        x = nn.functional.interpolate(
            x, size=low_level_features.shape[-2:], mode="bilinear", align_corners=False
        )

        x = torch.cat([x, low_level_features], dim=1)
        x = self.concat_conv(x)

        x = nn.functional.interpolate(
            x, size=input_shape, mode="bilinear", align_corners=False
        )

        return x


def get_fscnn_mobilenetv3_model(
    num_classes,
    device,
    pretrained_backbone=True,
    freeze_layers=None,
    unfreeze_layers=None,
):
    """
    Helper function to get the FSCNN_MobileNetV3 model.
    """
    model = FSCNN_MobileNetV3(
        num_classes=num_classes, pretrained_backbone=pretrained_backbone
    )

    if freeze_layers:
        for name, param in model.named_parameters():
            if any(fl in name for fl in freeze_layers):
                param.requires_grad = False

    if unfreeze_layers:
        for name, param in model.named_parameters():
            if any(ufl in name for ufl in unfreeze_layers):
                param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    for param in model.conv_low_level.parameters():
        param.requires_grad = True
    for param in model.concat_conv.parameters():
        param.requires_grad = True

    model.to(device)
    return model

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models._utils import IntermediateLayerGetter

# Conditional import for MobileNetV3Large_Weights for broader compatibility
try:
    from torchvision.models import MobileNet_V3_Large_Weights

    _use_new_weights_api = True
except ImportError:
    _use_new_weights_api = False
    print(
        "Warning: MobileNetV3Large_Weights not found. Using pretrained=True for backward compatibility."
    )


class FSCNN_MobileNetV3(nn.Module):
    """
    FSCNN-like architecture with MobileNetV3 as the backbone for semantic segmentation.
    This aims to be a simplified FCN/Deeplab-like structure using MobileNetV3 features.
    """

    def __init__(self, num_classes, pretrained_backbone=True):
        super(FSCNN_MobileNetV3, self).__init__()

        if pretrained_backbone:
            if _use_new_weights_api:
                weights = MobileNet_V3_Large_Weights.DEFAULT
                # Use pretrained=False and then load weights explicitly if DEFAULT is not available
                # or rely on pretrained=True if it works for your torchvision version
                # For simplicity, stick to pretrained=True for older versions and weights=DEFAULT for newer
                backbone = models.mobilenet_v3_large(weights=weights, dilated=True)
            else:
                backbone = models.mobilenet_v3_large(pretrained=True, dilated=True)
        else:
            backbone = models.mobilenet_v3_large(dilated=True)

        # Corrected way to get features for IntermediateLayerGetter
        # Pass backbone.features directly, and use integer indices for return_layers
        # because backbone.features is an nn.Sequential
        return_layers = {
            "12": "low_level",  # Corresponds to backbone.features[12]
            "16": "out",  # Corresponds to backbone.features[16]
        }

        self.backbone_features = IntermediateLayerGetter(
            backbone.features, return_layers=return_layers
        )  # <--- MODIFIED LINE HERE

        # The number of channels for 'out' and 'low_level' are fixed for MobileNetV3Large
        inplanes = 960  # Output channels of backbone.features[16]
        low_level_inplanes = 112  # Output channels of backbone.features[12]

        self.classifier = FCNHead(inplanes, num_classes)

        self.conv_low_level = nn.Conv2d(low_level_inplanes, 48, kernel_size=1)
        self.concat_conv = nn.Conv2d(
            num_classes + 48, num_classes, kernel_size=3, padding=1
        )

    def forward(self, x):
        input_shape = x.shape[-2:]
        # Get features from the IntermediateLayerGetter on the backbone.features
        features = self.backbone_features(
            x
        )  # <--- MODIFIED HERE (calling backbone_features)

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
    # print(model)

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

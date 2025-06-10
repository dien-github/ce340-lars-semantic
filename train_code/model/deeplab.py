import torch
import torchvision

def set_requires_grad(model, freeze_layers=None, unfreeze_layers=None):
    """
    Freeze or unfreeze layers by name or type.
    Args:
        model: nn.Module
        freeze_layers: list of str or type, layers to freeze (set requires_grad=False)
        unfreeze_layers: list of str or type, layers to unfreeze (set requires_grad=True)
    """
    if freeze_layers is None:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for name, module in model.named_modules():
            if any(
                (isinstance(module, t) if isinstance(t, type) else t in name)
                for t in freeze_layers
            ):
                for param in module.parameters(recurse=False):
                    param.requires_grad = False

    if unfreeze_layers is not None:
        for name, module in model.named_modules():
            if any(
                (isinstance(module, t) if isinstance(t, type) else t in name)
                for t in unfreeze_layers
            ):
                for param in module.parameters(recurse=False):
                    param.requires_grad = True

def get_deeplab_model(num_classes, device, freeze_layers=None, unfreeze_layers=None):
    """
    Load the pre-trained DeepLabV3 model with MobileNetV3 backbone.
    
    Args:
        num_classes (int): Number of classes for segmentation.
        device (torch.device): Device to load the model on.
        
    Returns:
        model (torch.nn.Module): The DeepLabV3 model.
    """
    pretrained_model_num_classes = 21

    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=pretrained_model_num_classes,
    )
    original_final_conv = model.classifier[4]
    model.classifier[4] = torch.nn.Conv2d(
        in_channels=original_final_conv.in_channels,
        out_channels=num_classes,
        kernel_size=original_final_conv.kernel_size,
        stride=original_final_conv.stride
    )
    model = model.to(device)
    
    set_requires_grad(
        model,
        freeze_layers=freeze_layers,
        unfreeze_layers=unfreeze_layers if unfreeze_layers is not None else ["classifier"]
    )
    return model

def get_lraspp_model(num_classes, device, freeze_layers=None, unfreeze_layers=None):
    """
    Load the pre-trained LR-ASPP model with MobileNetV3 backbone.
    """
    pretrained_model_num_classes = 21
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=pretrained_model_num_classes
    )
    original_low_conv = model.classifier.low_classifier
    original_high_conv = model.classifier.high_classifier
    model.classifier.low_classifier = torch.nn.Conv2d(
        in_channels=original_low_conv.in_channels,
        out_channels=num_classes,
        kernel_size=original_low_conv.kernel_size,
        stride=original_low_conv.stride
    )
    model.classifier.high_classifier = torch.nn.Conv2d(
        in_channels=original_high_conv.in_channels,
        out_channels=num_classes,
        kernel_size=original_high_conv.kernel_size,
        stride=original_high_conv.stride
    )
    model = model.to(device)

    set_requires_grad(
        model,
        freeze_layers=freeze_layers,
        unfreeze_layers=unfreeze_layers if unfreeze_layers is not None else ["classifier"]
    )
    return model
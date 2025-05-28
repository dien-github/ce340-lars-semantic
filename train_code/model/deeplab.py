import torch
import torchvision

def get_deeplab_model(num_classes, device):
    """
    Load the pre-trained DeepLabV3 model with MobileNetV3 backbone.
    
    Args:
        num_classes (int): Number of classes for segmentation.
        device (torch.device): Device to load the model on.
        
    Returns:
        model (torch.nn.Module): The DeepLabV3 model.
    """
    # When loading pretrained weights for the entire model, torchvision expects
    # the num_classes argument during initialization to either be None (to use the
    # pretrained head's class count) or to match the pretrained head's class count.
    # To fine-tune on a different number of classes, we first load the model
    # with a num_classes compatible with the weights (e.g., 21 for COCO default),
    # and then manually replace the final classification layer of the classifier head.
    
    # The default weights are for COCO, which has 21 classes.
    pretrained_model_num_classes = 21

    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        weights=torchvision.models.segmentation.DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=pretrained_model_num_classes, # Load with the number of classes of the pretrained head
    )
    # model.classifier is a DeepLabHead, which is an nn.Sequential.
    # The last layer (index 4) is the final classification Conv2d layer.
    # We replace it to match the desired number of output classes ('num_classes' argument of this function).
    original_final_conv = model.classifier[4]
    model.classifier[4] = torch.nn.Conv2d(
        in_channels=original_final_conv.in_channels, # This will be 256 for DeepLabV3+MobileNetV3
        out_channels=num_classes, # This is the target num_classes (e.g., 3 from config)
        kernel_size=original_final_conv.kernel_size,
        stride=original_final_conv.stride
    )
    model = model.to(device)
    
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model

def get_lraspp_model(num_classes, device):
    """
    Load the pre-trained LR-ASPP model with MobileNetV3 backbone.
    """
    pretrained_model_num_classes = 21
    model = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
        weights=torchvision.models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        num_classes=pretrained_model_num_classes
    )
    # Thay đổi cả hai classifier để cùng output đúng số class
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

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze only the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model
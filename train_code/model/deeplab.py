def get_deeplab_model(num_classes, device):
    """
    Load the pre-trained DeepLabV3 model with MobileNetV3 backbone.
    
    Args:
        num_classes (int): Number of classes for segmentation.
        device (torch.device): Device to load the model on.
        
    Returns:
        model (torch.nn.Module): The DeepLabV3 model.
    """
    model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(
        pretrained=True,
        num_classes=num_classes,
    )
    # model.classifier[4] = torch.nn.Conv2d(
    #     in_channels=960, 
    #     out_channels=num_classes, 
    #     kernel_size=(1, 1), 
    #     stride=(1, 1)
    # )
    model = model.to(device)
    
    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    return model
""" FCN-RESNET101 Model download and change the head for different output classes"""

from torchvision.models.segmentation.fcn import FCNHead
from torchvision import models


def createFCNResnet101(outputchannels=1, feature_extract=True):
    """FCNResnet101 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
        feature_extract (bool, optional): If False the whole model is
        trained otherwise only the classifier (head) is trained
    Returns:
        model: Returns the FCN model with the ResNet101 backbone.
    """
    model = models.segmentation.fcn_resnet101(pretrained=True,progress=True)
    print(model)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

    
    model.classifier = FCNHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model
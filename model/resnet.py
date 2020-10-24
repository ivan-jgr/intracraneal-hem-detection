import torch.nn as nn

from torchvision import models


class ResnetModel(nn.Module):

    def __init__(self, classes):
        """
        Arguments
        ---------
        classes: número de clases (tipos de hemorragias)
        """
        super(ResnetModel, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        n_filters = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(n_filters, classes)

    def forward(self, x):
        x = self.backbone(x)

        return x

import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained)
        self.vgg16 = nn.Sequential(*list(self.vgg16.features.children())[:-1])
        in_features = 512  # 输出通道数，因为VGG16最后一层是512个通道

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        features = self.vgg16(x)
        output = self.fc(features)
        return output

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(MobileNetV2, self).__init__()
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
        in_features = self.mobilenet_v2.classifier[1].in_features
        self.mobilenet_v2.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        output = self.mobilenet_v2(x)
        return output


class InceptionV3(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(InceptionV3, self).__init__()
        self.inception = models.inception_v3(pretrained=pretrained)
        in_features = self.inception.fc.in_features

        self.inception.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        features = self.inception(x)
        output = self.classifier(features)
        return output
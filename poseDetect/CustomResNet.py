import torch.nn as nn
from torchvision import models

class ResNet101(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet101, self).__init__()
        self.resnet = models.resnet101(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

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
        features = self.resnet(x)
        output = self.fc(features)
        return output

class ResNet34(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet34, self).__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 可以防止过拟合
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        output = self.fc(features)
        return output

class ResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 可以防止过拟合
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        output = self.fc(features)
        return output
class ResNet18(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        in_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout 可以防止过拟合
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.resnet(x)
        output = self.fc(features)
        return output
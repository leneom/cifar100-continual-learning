from __future__ import annotations

from torch import nn


class TinyConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, inputs):
        features = self.features(inputs)
        return self.classifier(features.flatten(1))


def cifar_resnet18(num_classes: int) -> nn.Module:
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "tiny":
        return TinyConvNet(num_classes)
    if name == "resnet18":
        return cifar_resnet18(num_classes)
    raise ValueError(f"unknown model: {name}")


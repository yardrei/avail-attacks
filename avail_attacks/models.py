import torch
from torch import nn
import torch.nn.functional as F


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()

        self.start_channels = 64
        # this is resnet-18, resnet-50 is [3, 4, 6, 3]
        self.number_of_blocks_list = [2, 2, 2, 2]

        self.conv = nn.Conv2d(
            3, self.start_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(self.start_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            self.start_channels, 64, self.number_of_blocks_list[0]
        )
        self.layer2 = self._make_layer(
            64, 128, self.number_of_blocks_list[1], 2)
        self.layer3 = self._make_layer(
            128, 256, self.number_of_blocks_list[2], 2)
        self.layer4 = self._make_layer(
            256, 512, self.number_of_blocks_list[3], 2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, number_of_blocks, stride=1):
        layers = [ResnetBlock(in_channels, out_channels, stride)]

        for i in range(1, number_of_blocks):
            layers.append(ResnetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out


class VGGCifar(nn.Module):
    def __init__(self, no_dropout: bool):
        super().__init__()
        layers = list()

        layers.extend(
            [
                # conv1
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                # conv2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, stride=2),
                # conv3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2, stride=2),
                # conv4
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2, stride=2),
                # conv5
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2, stride=2),
            ]
        )

        self.features = nn.Sequential(*layers)

        # for cifar-10 it's 1*1
        if no_dropout:
            classifier_layers = [nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 10), ]
        else:
            classifier_layers = [nn.Dropout(),
                                 nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 10), ]

        self.classifier = nn.Sequential(
            *classifier_layers
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


# class VGGMnist(nn.Module):
#     def __init__(self):
#         super().__init__()
#         layers = list()
#
#         layers.extend(
#             [
#                 # conv1
#                 nn.Conv2d(1, 64, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(64, 64, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, stride=2),
#                 # conv2
#                 nn.Conv2d(64, 128, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(128, 128, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, stride=2),
#                 # conv3
#                 nn.Conv2d(128, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, stride=2),
#                 # conv4
#                 nn.Conv2d(256, 512, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(512, 512, 3, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(512, 512, 3, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, stride=2),
#             ]
#         )
#
#         self.features = nn.Sequential(*layers)
#
#         self.classifier = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(512, 10),
#         )
#
#     def forward(self, x):
#         out = self.features(x)
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#
#         return out


class VGGMnist(nn.Module):
    def __init__(self, no_dropout: bool):
        super().__init__()
        layers = list()

        layers.extend(
            [
                # conv1
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, stride=2),
                # conv2
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, stride=2),
                # conv3
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(2, stride=2),
                # conv4
                nn.Conv2d(256, 512, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(2, stride=2),
            ]
        )

        self.features = nn.Sequential(*layers)
        if no_dropout:
            classifier_layers = [nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, 512),
                            nn.ReLU(),
                            nn.Linear(512, 10), ]
        else:
            classifier_layers = [nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(512, 512),
                                 nn.ReLU(),
                                 nn.Dropout(),
                                 nn.Linear(512, 10), ]

        self.classifier = nn.Sequential(
            *classifier_layers
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

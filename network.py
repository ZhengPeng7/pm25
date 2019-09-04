import torch
import torch.nn as nn
from torchvision import models

class Conv2D_BN_activa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, pad,
            dilation=1, if_bn=True, activation=nn.ReLU(inplace=True)
    ):
        super(Conv2D_BN_activa, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation)
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv2d(x)
        if self.if_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Encoder(nn.Module):
    def __init__(self, pretrain=True):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[0:3])
        self.conv2 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[4:8])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Image
        self.conv1 = Conv2D_BN_activa(512, 128, 3, 1, 1)
        self.conv2 = Conv2D_BN_activa(128, 64, 3, 1, 1)
        # TBV
        self.conv_TBV = Conv2D_BN_activa(1, 8, 3, 1, 1)
        # Entropy
        self.conv_ent = Conv2D_BN_activa(1, 8, 3, 1, 1)
        self.dense_1 = nn.Linear(100, 128)
        self.drop_1 = nn.Dropout(0.2)
        self.dense_2 = nn.Linear(128, 64)
        self.drop_2 = nn.Dropout(0.2)
        self.dense_3 = nn.Linear(64, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, image, TBV, entropy):
        image = self.conv1(image)
        image = self.conv2(image)

        TBV = self.conv_TBV(TBV)

        entropy = self.conv_ent(entropy)

        x = torch.cat([image, TBV, entropy], axis=0)
        print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.relu(self.dense_3(self.drop_2(self.dense_2(self.drop_1(self.dense_1(x))))))

        return x


class Network(nn.Module):
    def __init__(self, pretrain=True):
        super(Network, self).__init__()
        self.encoder = Encoder(pretrain=pretrain)
        self.decoder = Decoder()

    def forward(self, image, TBV, entropy):
        image = self.encoder(image)
        x = self.decoder(image, TBV, entropy)
        return x

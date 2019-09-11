import torch
import torch.nn as nn
from torchvision import models

class Conv2D_BN_activa(nn.Module):
    def __init__(
            self, in_channels, out_channels, kernel_size, stride, pad,
            dilation=1, if_bn=True, activation='relu'
    ):
        super(Conv2D_BN_activa, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, dilation=dilation, bias=(not if_bn))
        self.if_bn = if_bn
        if self.if_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)

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
        # self.conv1 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[0:3])
        # self.conv2 = nn.Sequential(*list(models.resnet50(pretrained=pretrain).children())[4:8])
        self.vgg_backbone = nn.Sequential(*list(models.vgg16_bn(pretrained=pretrain).children())[:-1])

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.conv2(x)
        x = self.vgg_backbone(x)
        # print('Shape of ouput of vgg_bb:', x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Image
        self.conv1 = Conv2D_BN_activa(512, 126, 1, 1, 0)
        # TBV
        self.conv_TBV = Conv2D_BN_activa(1, 1, 1, 1, 0)
        # Entropy
        self.conv_ent = Conv2D_BN_activa(1, 1, 1, 1, 0)
        self.dense_1 = nn.Linear(6272, 1024)
        self.relu_1 = nn.ReLU(inplace=True)
        self.drop_1 = nn.Dropout(0.2)
        self.dense_2 = nn.Linear(1024, 1024)
        self.relu_2 = nn.ReLU(inplace=True)
        self.drop_2 = nn.Dropout(0.2)
        self.dense_3 = nn.Linear(1024, 64)
        self.relu_3 = nn.ReLU(inplace=True)
        self.drop_3 = nn.Dropout(0.2)
        self.dense_4 = nn.Linear(64, 1)
        self.relu_4 = nn.ReLU(inplace=True)

    def forward(self, image, TBV, entropy):
        image = self.conv1(image)
        # print('image.shape =', image.shape)
        TBV = torch.zeros((image.shape[0], 1, image.shape[-2], image.shape[-1]), dtype=image.dtype).cuda() + TBV
        entropy = torch.zeros((image.shape[0], 1, image.shape[-2], image.shape[-1]), dtype=image.dtype).cuda() + entropy

        x = torch.cat([image, TBV, entropy], dim=1)
        x = x.view(x.size(0), -1)
        # print('Shape of x right before dense layers:', x.shape)

        x = self.dense_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)

        x = self.dense_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)

        x = self.dense_3(x)
        x = self.relu_3(x)
        x = self.drop_3(x)

        x = self.dense_4(x)
        x = self.relu_4(x)

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

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models

BatchNorm2d = nn.BatchNorm2d
nonlinearity = partial(F.relu, inplace=True)


class Bottleneck(nn.Module):  # 'resnet网络的基本框架’
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation,
                               bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):  # renet网络的构成部分
    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError
        # Modules
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3])
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1, 2, 4], stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, dilation=blocks[i] * dilation))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet101(nInputChannels=3, os=16, pretrained=False):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model


class ASPP_module(nn.Module):  # ASpp模块的组成
    def __init__(self, inplanes, planes, dilation):
        super(ASPP_module, self).__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPPPoolingH(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingH, self).__init__(
            nn.AdaptiveAvgPool2d((32, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPPPoolingW(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingW, self).__init__(
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# 正式开始deeplabv3+的结构组成
class Deeplabv3_plus(nn.Module):
    def get_name(self):
        return 'Deeplabv3_plus'

    def __init__(self, nInputChannels=3, n_classes=1, os=16, pretrained=False, freeze_bn=False, _print=True):
        super(Deeplabv3_plus, self).__init__()

        filters = [64, 64, 128, 256, 512]

        ## -------------Encoder--------------
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # ASPP,挑选参数
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        # 四个不同带洞卷积的设置，获取不同感受野
        self.aspp1 = ASPP_module(2048, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(2048, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(2048, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(2048, 256, dilation=dilations[3])
        self.relu = nn.ReLU()

        # 全局平均池化层的设置
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(2816, 256, 1, bias=False)
        self.bn1 = BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = BatchNorm2d(48)
        # 结构图中的解码部分的最后一个3*3的卷积块
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        if freeze_bn:
            self._freeze_bn()

        #     完成一些条带卷积的操作和条带池化的操作
        # 水平的扩展卷积
        self.dilater = nn.Conv2d(2048, 256, kernel_size=(3, 1), dilation=6, padding=(6, 0))
        # 竖直的扩展卷积，膨胀率1 2 4 8
        self.dilatec = nn.Conv2d(2048, 256, kernel_size=(1, 3), dilation=6, padding=(0, 6))

        # 竖直方向的平均池化
        self.ASPPH = ASPPPoolingH(in_channels=2048, out_channels=256)
        # 水平方向的平均池化
        self.ASPPW = ASPPPoolingW(in_channels=2048, out_channels=256)

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

    # 前向传播

    def forward(self, input):
        # x, low_level_features = self.resnet_features(input)
        # print("input:", input.shape)  # 3 1024 1024
        x = self.firstconv(input)
        # print("x:", x.shape)  # 64, 512, 512
        x = self.firstbn(x)
        # print("x:", x.shape)  # 64, 512, 512
        x = self.firstrelu(x)
        # print("x:", x.shape)  # 64, 512, 512
        x = self.firstmaxpool(x)
        # print("x:", x.shape)  # 64, 256, 256
        x = self.encoder1(x)
        # print("x:", x.shape)  # 256, 256, 256
        low_level_features = x
        x = self.encoder2(x)
        # print("x:", x.shape)  # 512, 128, 128
        x = self.encoder3(x)
        # print("x:", x.shape)  # 1024, 64, 64
        x = self.encoder4(x)
        # print(x.shape)  # [3, 2048, 32, 32]
        x1 = self.aspp1(x)
        # print("x1:", x1.shape)  # [3, 256, 32, 32]
        x2 = self.aspp2(x)
        # print("x2:", x2.shape)  # [3, 256, 32, 32]
        x3 = self.aspp3(x)
        # print("x3:", x3.shape)  # [3, 256, 32, 32]
        x4 = self.aspp4(x)
        # print("x4:", x4.shape)  # [3, 256, 32, 32]

        x5 = self.global_avg_pool(x)
        # print("x5:", x5.shape)  # [3, 256, 1, 1]
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # print("x5:", x5.shape)  # [3, 256, 32, 32]

        # 条带池化
        x6 = self.ASPPH(x)
        # print("x6:", x6.shape)  # [3, 256, 32, 32]
        x7 = self.ASPPW(x)
        # print("x7:", x7.shape)  # [3, 256, 32, 32]

        # 条带卷积
        # 水平
        x8 = nonlinearity(self.dilater(x))
        # print("x8:", x8.shape)  # [3, 256, 32, 32]
        # 竖直
        x9 = nonlinearity(self.dilatec(x))
        # print("x9:", x9.shape)  # [3, 256, 32, 32]
        # 正斜对角线
        x10 = nonlinearity(self.inv_h_transform(self.dilater(self.h_transform(x))))
        # print("x10:", x10.shape)  # [3, 256, 32, 32]
        # 反斜对角线
        x11 = nonlinearity(self.inv_h_transform(self.dilatec(self.h_transform(x))))
        # print("x11:", x11.shape)  # [3, 256, 32, 32]

        # 把四个ASPP模块以及全局池化层拼接起来 在拼接上两个条带池化，四个条带卷积
        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # # print("cat:", x.shape)  # [3, 1280, 32, 32]
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), dim=1)
        # print("cat:", x.shape)  # [3, 2816, 32, 32]

        # 上采样
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)

        x = self.relu(x)
        # print(x.shape)

        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)
        # print("xxxx:", x.shape)  # 256 256 256
        # print("low_level_features:", low_level_features.shape)  # 256, 256, 256
        low_level_features = self.conv2(low_level_features)
        # print("low_level_features:", low_level_features.shape)  # 48, 256, 256

        low_level_features = self.bn2(low_level_features)
        # print("low_level_features:", low_level_features.shape)  # 48, 256, 256

        low_level_features = self.relu(low_level_features)
        # print("low_level_features:", low_level_features.shape)  # 48, 256, 256

        # 拼接低层次的特征，然后再通过插值获取原图大小的结果
        x = torch.cat((x, low_level_features), dim=1)
        # print("x:", x.shape)  # 304, 256, 256
        x = self.last_conv(x)
        # print("x:", x.shape)  # 1, 256, 256
        # 实现插值和上采样
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # print("x:", x.shape)  # 1, 1024, 1024
        x = F.sigmoid(x)
        # print("x:", x.shape)  # 1, 1024, 1024
        return x

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k

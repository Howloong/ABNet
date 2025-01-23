import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

BatchNorm2d = nn.BatchNorm2d


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        # conv则是实际进行的卷积操作，注意这里步长设置为卷积核大小，因为与该卷积核进行卷积操作的特征图是由输出特征图中每个点扩展为其对应卷积核那么多个点后生成的。
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        # p_conv是生成offsets所使用的卷积，输出通道数为卷积核尺寸的平方的2倍，代表对应卷积核每个位置横纵坐标都有偏移量。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation  # modulation是可选参数,若设置为True,那么在进行卷积操作时,对应卷积核的每个位置都会分配一个权重。
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)  # 生成offsets所使用的卷积
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)  # 卷积核内相对于中心的偏移量坐标，前N个代表x坐标，后N个代表y坐标

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)  # reshape
        q_lt = p.detach().floor()
        # detach() 只训练部分分支网络，并不让其梯度对主网络的梯度造成影响，这时候我们就需要使用detach()函数来切断一些分支的反向传播
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        # 上下左右 四个方向
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        # 由于卷积核中心点位置是其尺寸的一半，于是中心点向左（上）方向移动尺寸的一半就得到起始点，向右（下）方向移动另一半就得到终止点
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        # p0_y、p0_x就是输出特征图每点映射到输入特征图上的纵、横坐标值。
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))

        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    # 输出特征图上每点（对应卷积核中心）加上其对应卷积核每个位置的相对（横、纵）坐标后再加上自学习的（横、纵坐标）偏移量。
    # p0就是将输出特征图每点对应到卷积核中心，然后映射到输入特征图中的位置；
    # pn则是p0对应卷积核每个位置的相对坐标；
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # 计算双线性插值点的4邻域点对应的权重
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset


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


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


# 正式开始ABNet的结构组成
class ABNet(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=1, os=16, pretrained=False, freeze_bn=False, _print=True):
        super(ABNet, self).__init__()

        filters = [64, 64, 128, 256, 512]

        ## -------------Encoder--------------
        super().__init__()
        resnet = models.resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = DecoderBlock(1280, 512)
        self.decoder2 = DecoderBlock(512, 256)

        # ASPP,挑选参数
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        # 四个不同带洞卷积的设置，获取不同感受野
        self.aspp1 = ASPP_module(512, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(512, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(512, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(512, 256, dilation=dilations[3])
        self.relu = nn.ReLU()

        # 全局平均池化层的设置
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(512, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(64, 48, 1, bias=False)
        self.bn2 = BatchNorm2d(48)

        # 可变卷积
        self.deformConv64_48 = DeformConv2d(inc=64, outc=48)
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

    # 前向传播
    def forward(self, input):
        x = self.firstconv(input)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        x = self.encoder1(x)
        low_level_features = x
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # 把四个ASPP模块以及全局池化层拼接起来
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # 上采样 两层解码器
        d2 = self.decoder1(x)
        d1 = self.decoder2(d2)
        d1 = F.upsample(d1, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        low_level_features = self.deformConv64_48(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        # 拼接低层次的特征，然后再通过插值获取原图大小的结果
        x = torch.cat((d1, low_level_features), dim=1)
        x = self.last_conv(x)
        # 实现插值和上采样
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        x = F.sigmoid(x)
        # print("x", x.shape)
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


if __name__ == "__main__":
    import numpy as np

    a = np.random.rand(2, 3, 1024, 1024)
    a = torch.from_numpy(a).to(torch.float32)
    RCFSNet = ABNet()
    predicted = RCFSNet(a)

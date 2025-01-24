import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from vit_seg_modeling import Transformer, DecoderCup

BatchNorm2d = nn.BatchNorm2d

nonlinearity = partial(F.relu, inplace=True)


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


class ADEFNet(nn.Module):
    def get_name(self):
        return 'ADEFNet'
    def __init__(self, n_classes=1, os=16, freeze_bn=False):

        super(ADEFNet, self).__init__()

        ## -------------Encoder--------------
        super().__init__()
        resnet = models.resnet34(pretrained=False)
        # resnet.load_state_dict(torch.load('./network_new/resnet34.pth'))
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        config = CONFIGS_ViT_seg['R50-ViT-B_16']
        config.patches.grid = (16, 16)
        self.transformer = Transformer(config, img_size=256, vis=False)
        self.decoder = DecoderCup(config)

        # ASPP,挑选参数
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        # 四个不同带洞卷积的设置，获取不同感受野
        self.aspp1 = ASPP_module(768, 256, dilation=dilations[0])
        self.aspp2 = ASPP_module(768, 256, dilation=dilations[1])
        self.aspp3 = ASPP_module(768, 256, dilation=dilations[2])
        self.aspp4 = ASPP_module(768, 256, dilation=dilations[3])
        self.relu = nn.ReLU()

        # 全局平均池化层的设置
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(768, 256, 1, stride=1, bias=False),
                                             BatchNorm2d(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(2816, 256, 1, bias=False)
        self.bn1 = BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(64, 48, 1, bias=False)
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
        self.dilater = nn.Conv2d(768, 256, kernel_size=(3, 1), dilation=6, padding=(6, 0))
        # 竖直的扩展卷积，膨胀率1 2 4 8
        self.dilatec = nn.Conv2d(768, 256, kernel_size=(1, 3), dilation=6, padding=(0, 6))

        # 竖直方向的平均池化
        self.ASPPH = ASPPPoolingH(in_channels=768, out_channels=256)
        # 水平方向的平均池化
        self.ASPPW = ASPPPoolingW(in_channels=768, out_channels=256)

        # 可变卷积
        self.deformConv64_48 = DeformConv2d(inc=64, outc=48)

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
        x = self.firstconv(input)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        x0 = x
        # print("pre", x.shape)  # 64, 256, 256
        x = self.encoder1(x)
        # print("encoder1", x.shape)  # 64, 256, 256
        low_level_features = x
        # print("low", low_level_features.shape)  # 64, 256, 256
        x = self.encoder2(x)
        # print("encoder2", x.shape)  # 128, 128, 128
        x = self.encoder3(x)
        # print("encoder3", x.shape)  # 256, 64, 64
        x = self.encoder4(x)
        # print("encoder4", x.shape)  # 512, 32, 32
        x_decoder, attn_weights, features = self.transformer(x0)  # 1024 768
        # print("x_decoder:",x_decoder.shape)
        x_trans = self.decoder(x_decoder, features)  # 256 32 32
        x = torch.cat((x, x_trans), dim=1)  # 768 32 32

        x1 = self.aspp1(x)
        # print("aspp1", x1.shape)  # 256, 32, 32
        x2 = self.aspp2(x)
        # print("aspp2:", x2.shape)  # 256, 32, 32
        x3 = self.aspp3(x)
        # print("aspp3:", x3.shape)  # 256, 32, 32
        x4 = self.aspp4(x)
        # print("aspp4:", x4.shape)  # 256, 32, 32

        x5 = self.global_avg_pool(x)
        # print("GAP:", x5.shape)  # 256, 1, 1
        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        # print("x5:", x5.shape)  # 256, 32, 32

        # 条带池化
        x6 = self.ASPPH(x)
        # print("ASPPH:", x6.shape)  # 256, 32, 32
        x7 = self.ASPPW(x)
        # print("ASPPW:", x7.shape)  # 256, 32, 32

        # 条带卷积
        # 水平
        x8 = nonlinearity(self.dilater(x))
        # print("shuiping:", x8.shape)  # 256, 32, 32
        # 竖直
        x9 = nonlinearity(self.dilatec(x))
        # print("shuzhi:", x9.shape)  # 256, 32, 32
        # 正斜对角线
        x10 = nonlinearity(self.inv_h_transform(self.dilater(self.h_transform(x))))
        # print("zheng:", x10.shape)  # 256, 32, 32
        # 反斜对角线
        x11 = nonlinearity(self.inv_h_transform(self.dilatec(self.h_transform(x))))
        # print("fan:", x11.shape)  # 256, 32, 32

        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), dim=1)
        # print("cat:", x.shape)  # 2816, 32, 32

        # 上采样
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("1*1", x.shape)  # 256, 32, 32

        x = F.upsample(x, size=(int(math.ceil(input.size()[-2] / 4)),
                                int(math.ceil(input.size()[-1] / 4))), mode='bilinear', align_corners=True)
        # print("upsample", x.shape)  # 256, 256, 256
        low_level_features = self.deformConv64_48(low_level_features)
        # print("dc", low_level_features.shape)  # 48, 256, 256
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)

        # 拼接低层次的特征，然后再通过插值获取原图大小的结果
        x = torch.cat((x, low_level_features), dim=1)
        # print("dcatg", x.shape)  # 304, 256, 256
        x = self.last_conv(x)
        # print("lastconv", x.shape)  # 1, 256, 256
        # 实现插值和上采样
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # print("up", x.shape)  # 1, 1024, 1024
        x = F.sigmoid(x)
        # print("final", x.shape)  # 1, 1024, 1024
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

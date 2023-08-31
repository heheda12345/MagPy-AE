import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from collections import OrderedDict
import torchvision.transforms as transforms
import math
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction


class Concat(nn.Sequential):

    def __init__(self, *kargs, **kwargs):
        super(Concat, self).__init__(*kargs, **kwargs)

    def forward(self, inputs):
        return torch.cat([m(inputs) for m in self._modules.values()], 1)


class block(nn.Module):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block, self).__init__()
        self.scale = scale
        self.activation = activation or (lambda x: x)

    def forward(self, inputs):
        branch0 = self.Branch_0(inputs)
        branch1 = self.Branch_1(inputs)
        if hasattr(self, 'Branch_2'):
            branch2 = self.Branch_2(inputs)
            tower_mixed = torch.cat([branch0, branch1, branch2], 1)
        else:
            tower_mixed = torch.cat([branch0, branch1], 1)
        tower_out = self.Conv2d_1x1(tower_mixed)
        output = self.activation(self.scale * tower_out + inputs)
        return output


def conv_bn(in_planes, out_planes, kernel_size, stride=1, padding=0):
    """convolution with batchnorm, relu"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False), nn.BatchNorm2d(out_planes), nn.ReLU())


class block35(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block35, self).__init__(in_planes, scale, activation)
        self.Branch_0 = nn.Sequential(OrderedDict([('Conv2d_1x1', conv_bn(in_planes, 32, 1))]))
        self.Branch_1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 32, 1)), ('Conv2d_0b_3x3', conv_bn(32, 32, 3, padding=1))]))
        self.Branch_2 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 32, 1)), ('Conv2d_0b_3x3', conv_bn(32, 48, 3, padding=1)), ('Conv2d_0c_3x3', conv_bn(48, 64, 3, padding=1))]))
        self.Conv2d_1x1 = conv_bn(128, in_planes, 1)


class block17(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block17, self).__init__(in_planes, scale, activation)
        self.Branch_0 = nn.Sequential(OrderedDict([('Conv2d_1x1', conv_bn(in_planes, 192, 1))]))
        self.Branch_1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 128, 1)), ('Conv2d_0b_1x7', conv_bn(128, 160, (1, 7), padding=(0, 3))), ('Conv2d_0c_7x1', conv_bn(160, 192, (7, 1), padding=(3, 0)))]))
        self.Conv2d_1x1 = conv_bn(384, in_planes, 1)


class block8(block):

    def __init__(self, in_planes, scale=1.0, activation=nn.ReLU(True)):
        super(block8, self).__init__(in_planes, scale, activation)
        self.Branch_0 = nn.Sequential(OrderedDict([('Conv2d_1x1', conv_bn(in_planes, 192, 1))]))
        self.Branch_1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(in_planes, 192, 1)), ('Conv2d_0b_1x7', conv_bn(192, 224, (1, 3), padding=(0, 1))), ('Conv2d_0c_7x1', conv_bn(224, 256, (3, 1), padding=(1, 0)))]))
        self.Conv2d_1x1 = conv_bn(448, in_planes, 1)


class InceptionResnetV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(InceptionResnetV2, self).__init__()
        self.end_points = {}
        self.num_classes = num_classes
        self.stem = nn.Sequential(OrderedDict([('Conv2d_1a_3x3', conv_bn(3, 32, 3, stride=2, padding=1)), ('Conv2d_2a_3x3', conv_bn(32, 32, 3, padding=1)), ('Conv2d_2b_3x3', conv_bn(32, 64, 3)), ('MaxPool_3a_3x3', nn.MaxPool2d(3, 2)), ('Conv2d_3b_1x1', conv_bn(64, 80, 1)), ('Conv2d_4a_3x3', conv_bn(80, 192, 3)), ('MaxPool_5a_3x3', nn.MaxPool2d(3, 2))]))
        tower_conv = nn.Sequential(OrderedDict([('Conv2d_5b_b0_1x1', conv_bn(192, 96, 1))]))
        tower_conv1 = nn.Sequential(OrderedDict([('Conv2d_5b_b1_0a_1x1', conv_bn(192, 48, 1)), ('Conv2d_5b_b1_0b_5x5', conv_bn(48, 64, 5, padding=2))]))
        tower_conv2 = nn.Sequential(OrderedDict([('Conv2d_5b_b2_0a_1x1', conv_bn(192, 64, 1)), ('Conv2d_5b_b2_0b_3x3', conv_bn(64, 96, 3, padding=1)), ('Conv2d_5b_b2_0c_3x3', conv_bn(96, 96, 3, padding=1))]))
        tower_pool3 = nn.Sequential(OrderedDict([('AvgPool_5b_b3_0a_3x3', nn.AvgPool2d(3, stride=1, padding=1)), ('Conv2d_5b_b3_0b_1x1', conv_bn(192, 64, 1))]))
        self.mixed_5b = Concat(OrderedDict([('Branch_0', tower_conv), ('Branch_1', tower_conv1), ('Branch_2', tower_conv2), ('Branch_3', tower_pool3)]))
        self.blocks35 = nn.Sequential()
        for i in range(10):
            self.blocks35.add_module('Block35.%s' % i, block35(320, scale=0.17))
        tower_conv = nn.Sequential(OrderedDict([('Conv2d_6a_b0_0a_3x3', conv_bn(320, 384, 3, stride=2))]))
        tower_conv1 = nn.Sequential(OrderedDict([('Conv2d_6a_b1_0a_1x1', conv_bn(320, 256, 1)), ('Conv2d_6a_b1_0b_3x3', conv_bn(256, 256, 3, padding=1)), ('Conv2d_6a_b1_0c_3x3', conv_bn(256, 384, 3, stride=2))]))
        tower_pool = nn.Sequential(OrderedDict([('MaxPool_1a_3x3', nn.MaxPool2d(3, stride=2))]))
        self.mixed_6a = Concat(OrderedDict([('Branch_0', tower_conv), ('Branch_1', tower_conv1), ('Branch_2', tower_pool)]))
        self.blocks17 = nn.Sequential()
        for i in range(20):
            self.blocks17.add_module('Block17.%s' % i, block17(1088, scale=0.1))
        tower_conv = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(1088, 256, 1)), ('Conv2d_1a_3x3', conv_bn(256, 384, 3, stride=2))]))
        tower_conv1 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(1088, 256, 1)), ('Conv2d_1a_3x3', conv_bn(256, 64, 3, stride=2))]))
        tower_conv2 = nn.Sequential(OrderedDict([('Conv2d_0a_1x1', conv_bn(1088, 256, 1)), ('Conv2d_0b_3x3', conv_bn(256, 288, 3, padding=1)), ('Conv2d_1a_3x3', conv_bn(288, 320, 3, stride=2))]))
        tower_pool3 = nn.Sequential(OrderedDict([('MaxPool_1a_3x3', nn.MaxPool2d(3, stride=2))]))
        self.mixed_7a = Concat(OrderedDict([('Branch_0', tower_conv), ('Branch_1', tower_conv1), ('Branch_2', tower_conv2), ('Branch_3', tower_pool3)]))
        self.blocks8 = nn.Sequential()
        for i in range(9):
            self.blocks8.add_module('Block8.%s' % i, block8(1856, scale=0.2))
        self.blocks8.add_module('Block8.9', block8(1856, scale=0.2, activation=None))
        self.conv_pool = nn.Sequential(OrderedDict([('Conv2d_7b_1x1', conv_bn(1856, 1536, 1)), ('AvgPool_1a_8x8', nn.AvgPool2d(8, 1)), ('Dropout', nn.Dropout(0.2))]))
        self.classifier = nn.Linear(1536, num_classes)
        self.aux_classifier = nn.Sequential(OrderedDict([('Conv2d_1a_3x3', nn.AvgPool2d(5, 3)), ('Conv2d_1b_1x1', conv_bn(1088, 128, 1)), ('Conv2d_2a_5x5', conv_bn(128, 768, 5)), ('Dropout', nn.Dropout(0.2)), ('Logits', conv_bn(768, num_classes, 1))]))


        class aux_loss(nn.Module):

            def __init__(self):
                super(aux_loss, self).__init__()
                self.loss = nn.CrossEntropyLoss()

            def forward(self, outputs, target):
                return self.loss(outputs[0], target) + 0.4 * self.loss(outputs[1], target)
        self.criterion = aux_loss
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 90, 'lr': 0.0001}]

    def forward(self, x):
        x = self.stem(x)
        x = self.mixed_5b(x)
        x = self.blocks35(x)
        x = self.mixed_6a(x)
        branch1 = self.blocks17(x)
        x = self.mixed_7a(branch1)
        x = self.blocks8(x)
        x = self.conv_pool(x)
        x = x.view(-1, 1536)
        output = self.classifier(x)
        if hasattr(self, 'aux_classifier'):
            branch1 = self.aux_classifier(branch1).view(-1, self.num_classes)
            output = [output, branch1]
        return output


class InceptionModule(nn.Module):

    def __init__(self, in_channels, n1x1_channels, n3x3r_channels, n3x3_channels, dn3x3r_channels, dn3x3_channels, pool_proj_channels=None, type_pool='avg', stride=1):
        super(InceptionModule, self).__init__()
        self.in_channels = in_channels
        self.n1x1_channels = n1x1_channels or 0
        pool_proj_channels = pool_proj_channels or 0
        self.stride = stride
        if n1x1_channels > 0:
            self.conv_1x1 = conv_bn(in_channels, n1x1_channels, 1, stride)
        else:
            self.conv_1x1 = None
        self.conv_3x3 = nn.Sequential(conv_bn(in_channels, n3x3r_channels, 1), conv_bn(n3x3r_channels, n3x3_channels, 3, stride, padding=1))
        self.conv_d3x3 = nn.Sequential(conv_bn(in_channels, dn3x3r_channels, 1), conv_bn(dn3x3r_channels, dn3x3_channels, 3, padding=1), conv_bn(dn3x3_channels, dn3x3_channels, 3, stride, padding=1))
        if type_pool == 'avg':
            self.pool = nn.AvgPool2d(3, stride, padding=1)
        elif type_pool == 'max':
            self.pool = nn.MaxPool2d(3, stride, padding=1)
        if pool_proj_channels > 0:
            self.pool = nn.Sequential(self.pool, conv_bn(in_channels, pool_proj_channels, 1))

    def forward(self, inputs):
        layer_outputs = []
        if self.conv_1x1 is not None:
            layer_outputs.append(self.conv_1x1(inputs))
        layer_outputs.append(self.conv_3x3(inputs))
        layer_outputs.append(self.conv_d3x3(inputs))
        layer_outputs.append(self.pool(inputs))
        output = torch.cat(layer_outputs, 1)
        return output


def inception_v2(**kwargs):
    num_classes = getattr(kwargs, 'num_classes', 1000)
    return Inception_v2(num_classes=num_classes)


class Inception_v2(nn.Module):

    def __init__(self, num_classes=1000, aux_classifiers=True):
        super(inception_v2, self).__init__()
        self.num_classes = num_classes
        self.part1 = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.MaxPool2d(3, 2), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64, 192, 3, 1, 1, bias=False), nn.MaxPool2d(3, 2), nn.BatchNorm2d(192), nn.ReLU(), InceptionModule(192, 64, 64, 64, 64, 96, 32, 'avg'), InceptionModule(256, 64, 64, 96, 64, 96, 64, 'avg'), InceptionModule(320, 0, 128, 160, 64, 96, 0, 'max', 2))
        self.part2 = nn.Sequential(InceptionModule(576, 224, 64, 96, 96, 128, 128, 'avg'), InceptionModule(576, 192, 96, 128, 96, 128, 128, 'avg'), InceptionModule(576, 160, 128, 160, 128, 160, 96, 'avg'))
        self.part3 = nn.Sequential(InceptionModule(576, 96, 128, 192, 160, 192, 96, 'avg'), InceptionModule(576, 0, 128, 192, 192, 256, 0, 'max', 2), InceptionModule(1024, 352, 192, 320, 160, 224, 128, 'avg'), InceptionModule(1024, 352, 192, 320, 192, 224, 128, 'max'))
        self.main_classifier = nn.Sequential(nn.AvgPool2d(7, 1), nn.Dropout(0.2), nn.Conv2d(1024, self.num_classes, 1))
        if aux_classifiers:
            self.aux_classifier1 = nn.Sequential(nn.AvgPool2d(5, 3), conv_bn(576, 128, 1), conv_bn(128, 768, 4), nn.Dropout(0.2), nn.Conv2d(768, self.num_classes, 1))
            self.aux_classifier2 = nn.Sequential(nn.AvgPool2d(5, 3), conv_bn(576, 128, 1), conv_bn(128, 768, 4), nn.Dropout(0.2), nn.Conv2d(768, self.num_classes, 1))
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 90, 'lr': 0.0001}]


        class aux_loss(nn.Module):

            def __init__(self):
                super(aux_loss, self).__init__()
                self.loss = nn.CrossEntropyLoss()

            def forward(self, outputs, target):
                return self.loss(outputs[0], target) + 0.4 * (self.loss(outputs[1], target) + self.loss(outputs[2], target))
        self.criterion = aux_loss

    def forward(self, inputs):
        branch1 = self.part1(inputs)
        branch2 = self.part2(branch1)
        branch3 = self.part3(branch1)
        output = self.main_classifier(branch3).view(-1, self.num_classes)
        if hasattr(self, 'aux_classifier1'):
            branch1 = self.aux_classifier1(branch1).view(-1, self.num_classes)
            branch2 = self.aux_classifier2(branch2).view(-1, self.num_classes)
            output = [output, branch1, branch2]
        return output


class mnist_model(nn.Module):

    def __init__(self):
        super(mnist_model, self).__init__()
        self.feats = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.BatchNorm2d(32), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(64), nn.Conv2d(64, 64, 3, 1, 1), nn.MaxPool2d(2, 2), nn.ReLU(True), nn.BatchNorm2d(64), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.BatchNorm2d(128))
        self.classifier = nn.Conv2d(128, 10, 1)
        self.avgpool = nn.AvgPool2d(6, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = self.avgpool(out)
        out = out.view(-1, 10)
        return out


BIPRECISION = True


NUM_BITS = 8


NUM_BITS_GRAD = 8


NUM_BITS_WEIGHT = 8


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=False, inplace=False, enforce_true_zero=False, num_chunks=None, out_half=False):
        num_chunks = num_chunks = input.shape[0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(B // num_chunks, -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)
        if max_value is None:
            max_value = y.max(-1)[0].mean(-1)
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        qmin = 0.0
        qmax = 2.0 ** num_bits - 1.0
        scale = (max_value - min_value) / (qmax - qmin)
        scale = max(scale, 1e-08)
        if enforce_true_zero:
            initial_zero_point = qmin - min_value / scale
            zero_point = 0.0
            if initial_zero_point < qmin:
                zero_point = qmin
            elif initial_zero_point > qmax:
                zero_point = qmax
            else:
                zero_point = initial_zero_point
            zero_point = int(zero_point)
            output.div_(scale).add_(zero_point)
        else:
            output.add_(-min_value).div_(scale).add_(qmin)
        if ctx.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        output.clamp_(qmin, qmax).round_()
        if enforce_true_zero:
            output.add_(-zero_point).mul_(scale)
        else:
            output.add_(-qmin).mul_(scale).add_(min_value)
        if out_half and num_bits <= 16:
            output = output.half()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, num_bits=8, momentum=0.1):
        super(QuantMeasure, self).__init__()
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits

    def forward(self, input):
        if self.training:
            min_value = input.detach().view(input.size(0), -1).min(-1)[0].mean()
            max_value = input.detach().view(input.size(0), -1).max(-1)[0].mean()
            self.running_min.mul_(self.momentum).add_(min_value * (1 - self.momentum))
            self.running_max.mul_(self.momentum).add_(max_value * (1 - self.momentum))
        else:
            min_value = self.running_min
            max_value = self.running_max
        return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value), num_chunks=16)


class UniformQuantizeGrad(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.min_value is None:
            min_value = float(grad_output.min())
        else:
            min_value = ctx.min_value
        if ctx.max_value is None:
            max_value = float(grad_output.max())
        else:
            max_value = ctx.max_value
        grad_input = UniformQuantize().apply(grad_output, ctx.num_bits, min_value, max_value, ctx.stochastic, ctx.inplace)
        return grad_input, None, None, None, None, None


def quantize_grad(x, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
    return UniformQuantizeGrad().apply(x, num_bits, min_value, max_value, stochastic, inplace)


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias, stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None, stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


class QConv2d(nn.Conv2d):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.biprecision = biprecision

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight, min_value=float(self.weight.min()), max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.conv2d(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = conv2d_biprec(qinput, qweight, qbias, self.stride, self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        return output


class RangeBN(nn.Module):

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-05, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))
        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x)
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()
            y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)
            mean_min = y.min(-1)[0].mean(-1)
            mean = y.view(C, -1).mean(-1)
            scale_fix = 0.5 * 0.35 * (1 + (math.pi * math.log(4)) ** 0.5) / (2 * math.log(y.size(-1))) ** 0.5
            scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)
            self.running_mean.detach().mul_(self.momentum).add_(mean * (1 - self.momentum))
            self.running_var.detach().mul_(self.momentum).add_(scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        scale = quantize(scale, num_bits=self.num_bits, min_value=float(scale.min()), max_value=float(scale.max()))
        out = (x - mean.view(1, mean.size(0), 1, 1)) * scale.view(1, scale.size(0), 1, 1)
        if self.weight is not None:
            qweight = quantize(self.weight, num_bits=self.num_bits, min_value=float(self.weight.min()), max_value=float(self.weight.max()))
            out = out * qweight.view(1, qweight.size(0), 1, 1)
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias.view(1, qbias.size(0), 1, 1)
        if self.num_bits_grad is not None:
            out = quantize_grad(out, num_bits=self.num_bits_grad)
        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


class DepthwiseSeparableFusedConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableFusedConv2d, self).__init__()
        self.components = nn.Sequential(QConv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION), RangeBN(in_channels, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD), nn.ReLU(), QConv2d(in_channels, out_channels, 1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION), RangeBN(out_channels, num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD), nn.ReLU())

    def forward(self, x):
        return self.components(x)


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach() if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)

    def forward(self, input):
        qinput = self.quantize_input(input)
        qweight = quantize(self.weight, num_bits=self.num_bits_weight, min_value=float(self.weight.min()), max_value=float(self.weight.max()))
        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits_weight)
        else:
            qbias = None
        if not self.biprecision or self.num_bits_grad is None:
            output = F.linear(qinput, qweight, qbias)
            if self.num_bits_grad is not None:
                output = quantize_grad(output, num_bits=self.num_bits_grad)
        else:
            output = linear_biprec(qinput, qweight, qbias, self.num_bits_grad)
        return output


def nearby_int(n):
    return int(round(n))


class MobileNet(nn.Module):

    def __init__(self, width=1.0, shallow=False, num_classes=1000):
        super(MobileNet, self).__init__()
        num_classes = num_classes or 1000
        width = width or 1.0
        layers = [QConv2d(3, nearby_int(width * 32), kernel_size=3, stride=2, padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION), RangeBN(nearby_int(width * 32), num_bits=NUM_BITS, num_bits_grad=NUM_BITS_GRAD), nn.ReLU(inplace=True), DepthwiseSeparableFusedConv2d(nearby_int(width * 32), nearby_int(width * 64), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 64), nearby_int(width * 128), kernel_size=3, stride=2, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 128), nearby_int(width * 128), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 128), nearby_int(width * 256), kernel_size=3, stride=2, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 256), nearby_int(width * 256), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 256), nearby_int(width * 512), kernel_size=3, stride=2, padding=1)]
        if not shallow:
            layers += [DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 512), kernel_size=3, padding=1)]
        layers += [DepthwiseSeparableFusedConv2d(nearby_int(width * 512), nearby_int(width * 1024), kernel_size=3, stride=2, padding=1), DepthwiseSeparableFusedConv2d(nearby_int(width * 1024), nearby_int(width * 1024), kernel_size=3, stride=1, padding=1)]
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = QLinear(nearby_int(width * 1024), num_classes, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.input_transform = {'train': transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.3, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]), 'eval': transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])}
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001}, {'epoch': 80, 'lr': 0.0001}]

    @staticmethod
    def regularization(model, weight_decay=4e-05):
        l2_params = 0
        for m in model.modules():
            if isinstance(m, QConv2d) or isinstance(m, nn.Linear):
                l2_params += m.weight.pow(2).sum()
                if m.bias is not None:
                    l2_params += m.bias.pow(2).sum()
        return weight_decay * 0.5 * l2_params

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BiReLUFunction(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, inplace=False):
        if input.size(1) % 2 != 0:
            raise RuntimeError('dimension 1 of input must be multiple of 2, but got {}'.format(input.size(1)))
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        pos, neg = output.chunk(2, dim=1)
        pos.clamp_(min=0)
        neg.clamp_(max=0)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_variables
        grad_input = grad_output.masked_fill(output.eq(0), 0)
        return grad_input, None


def birelu(x, inplace=False):
    return BiReLUFunction().apply(x, inplace)


class BiReLU(nn.Module):
    """docstring for BiReLU."""

    def __init__(self, inplace=False):
        super(BiReLU, self).__init__()
        self.inplace = inplace

    def forward(self, inputs):
        return birelu(inputs, inplace=self.inplace)


def rnlu(x, inplace=False, shift=0, scale_fix=(math.pi / 2) ** 0.5):
    x = birelu(x, inplace=inplace)
    pos, neg = (x + shift).chunk(2, dim=1)
    scale = (pos - neg).view(pos.size(0), -1).mean(1) * scale_fix + 1e-08
    return x / scale.view(scale.size(0), *([1] * (x.dim() - 1)))


class RnLU(nn.Module):
    """docstring for RnLU."""

    def __init__(self, inplace=False):
        super(RnLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return rnlu(x, inplace=self.inplace)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def depBatchNorm2d(exists, *kargs, **kwargs):
    if exists:
        return nn.BatchNorm2d(*kargs, **kwargs)
    else:
        return lambda x: x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, batch_norm=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=not batch_norm)
        self.bn2 = depBatchNorm2d(batch_norm, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, batch_norm=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=not batch_norm, groups=32)
        self.bn2 = depBatchNorm2d(batch_norm, planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=not batch_norm)
        self.bn3 = depBatchNorm2d(batch_norm, planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(QConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class ResNet_imagenet(ResNet):

    def __init__(self, num_classes=1000, block=Bottleneck, layers=[3, 4, 23, 3]):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = 64
        self.conv1 = QConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = QLinear(512 * block.expansion, num_classes, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        init_model(self)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 90, 'lr': 0.0001}]


class ResNet_cifar10(ResNet):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18):
        super(ResNet_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = QConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = QLinear(64, num_classes, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD)
        init_model(self)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 81, 'lr': 0.01}, {'epoch': 122, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 164, 'lr': 0.0001}]


class PlainDownSample(nn.Module):

    def __init__(self, input_dims, output_dims, stride):
        super(PlainDownSample, self).__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.stride = stride
        self.downsample = nn.AvgPool2d(stride)
        self.zero = Variable(torch.Tensor(1, 1, 1, 1), requires_grad=False)

    def forward(self, inputs):
        ds = self.downsample(inputs)
        zeros_size = [ds.size(0), self.output_dims - ds.size(1), ds.size(2), ds.size(3)]
        return torch.cat([ds, self.zero.expand(*zeros_size)], 1)


class ResNeXt(nn.Module):

    def __init__(self, shortcut='B'):
        super(ResNeXt, self).__init__()
        self.shortcut = shortcut

    def _make_layer(self, block, planes, blocks, stride=1, batch_norm=True):
        downsample = None
        if self.shortcut == 'C' or self.shortcut == 'B' and (stride != 1 or self.inplanes != planes * block.expansion):
            downsample = [nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=not batch_norm)]
            if batch_norm:
                downsample.append(nn.BatchNorm2d(planes * block.expansion))
            downsample = nn.Sequential(*downsample)
        else:
            downsample = PlainDownSample(self.inplanes, planes * block.expansion, stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, batch_norm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, batch_norm=batch_norm))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNeXt_imagenet(ResNeXt):

    def __init__(self, num_classes=1000, block=Bottleneck, layers=[3, 4, 23, 3], batch_norm=True, shortcut='B'):
        super(ResNeXt_imagenet, self).__init__(shortcut=shortcut)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, batch_norm=batch_norm)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, batch_norm=batch_norm)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, batch_norm=batch_norm)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)
        init_model(self)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 30, 'lr': 0.01}, {'epoch': 60, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 90, 'lr': 0.0001}]


class ResNeXt_cifar10(ResNeXt):

    def __init__(self, num_classes=10, block=BasicBlock, depth=18, batch_norm=True):
        super(ResNeXt_cifar10, self).__init__()
        self.inplanes = 16
        n = int((depth - 2) / 6)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=not batch_norm)
        self.bn1 = depBatchNorm2d(batch_norm, 16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x
        self.layer1 = self._make_layer(block, 16, n, batch_norm=not batch_norm)
        self.layer2 = self._make_layer(block, 32, n, stride=2, batch_norm=not batch_norm)
        self.layer3 = self._make_layer(block, 64, n, stride=2, batch_norm=not batch_norm)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
        init_model(self)
        self.regime = [{'epoch': 0, 'optimizer': 'SGD', 'lr': 0.1, 'weight_decay': 0.0001, 'momentum': 0.9}, {'epoch': 81, 'lr': 0.01}, {'epoch': 122, 'lr': 0.001, 'weight_decay': 0}, {'epoch': 164, 'lr': 0.0001}]


def get_model():
    # ResNeXt_imagenet has lower speedup
    return ResNet_imagenet().cuda()


def get_input(batch_size):
    return (torch.randn(batch_size, 3, 224, 224).cuda(),), {}


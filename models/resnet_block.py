import torch
import common.torch


class ResNetBlock(torch.nn.Module):
    """
    ResNet block taken from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py.
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None, normalization=True):
        """
        Constructor.

        :param inplanes: input channels
        :type inplanes: int
        :param planes: output channels
        :type planes: int
        :param stride: stride
        :type stride: int
        :param downsample: whether to downsample
        :type downsample: bool
        :param normalization: whether to use normalization
        :type normalization: bool
        """

        super(ResNetBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.normalization = normalization
        if self.normalization:
            self.norm1 = torch.nn.BatchNorm2d(planes)
            torch.nn.init.constant_(self.norm1.weight, 1)
            torch.nn.init.constant_(self.norm1.bias, 0)

        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')

        if self.normalization:
            self.norm2 = torch.nn.BatchNorm2d(planes)
            torch.nn.init.constant_(self.norm2.weight, 1)
            torch.nn.init.constant_(self.norm2.bias, 0)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass.

        :param x: input
        :type x: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        out = self.conv1(x)
        if self.normalization:
            out = self.norm1(out)

        out = self.relu(out)
        out = self.conv2(out)
        if self.normalization:
            out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out += identity
        out = self.relu(out)

        return out
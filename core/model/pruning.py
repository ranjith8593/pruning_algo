"""Util file for modified Linear and Conv2d implementation
"""
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def to_var(x, requires_grad=False, volatile=False):
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


class ModifiedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(ModifiedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class ModifiedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModifiedConv2d, self).__init__(in_channels, out_channels, kernel_size)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.conv2d(x, weight, self.bias)
        else:
            return F.conv2d(x, self.weight, self.bias)

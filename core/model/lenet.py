import torch
import torch.nn as nn

from pruning import ModifiedLinear, ModifiedConv2d

class LeNet(nn.Module):
    def __init__(self, mask=None):
        super(LeNet, self).__init__()
        self.conv1 =  ModifiedConv2d(1,6,5)
        self.conv2 =  ModifiedConv2d(6,16,5)
        self.lin1 = ModifiedLinear(16 * 5 * 5, 120)
        self.lin2 = ModifiedLinear(120, 84)
        self.lin3 = ModifiedLinear(84,10)
        self.features = nn.Sequential(self.conv1,
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2),
                                      self.conv2,
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2)
                                      )
        self.classifier = nn.Sequential(self.lin1,
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.3),
                                        self.lin2,
                                        nn.ReLU(inplace=True),
                                        self.lin3
                                        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def set_mask(self, mask):
      self.conv1.set_mask(mask[0])
      self.conv2.set_mask(mask[1])
      self.lin1.set_mask(mask[2])
      self.lin2.set_mask(mask[3])
      self.lin3.set_mask(mask[4])
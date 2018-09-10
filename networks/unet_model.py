#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

import torch
import torch.nn as nn
import torch.nn.functional as F

# python 3 confusing imports :(
from networks.unet_parts import *
from networks.utils import  *
from torch.autograd import Variable
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(512,512, 256)
        self.up2 = up(256,256, 128)
        self.up3 = up(128,128, 64)
        self.up4 = up(64,64, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

    def get_net_name(self):
        return 'unet'
    def adaptive_bn(self,if_enable=False):
        if if_enable:
            for name,module  in self.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        #if 'down' in name or 'up' in name or 'inc' in name:
                            module.train()
                            module.track_running_stats=True
    def init_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d):
                print (module)

                #if 'down' in name or 'up' in name or 'inc' in name:
                #module.reset_parameters()
                    # print(name, module)
                module.running_mean.zero_()
                module.running_var.fill_(1)


class Deep_UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Deep_UNet, self).__init__()
        self.inc = inconv(n_channels, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512,512)
        self.up1 = up(512,512,512)
        self.up2 = up(512,512, 256)
        self.up3 = up(256,256, 128)
        self.up4 = up(128,128, 64)
        self.up5 = up(64,64, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        print ('input{}'.format(x.size()))
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6,x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x
    def predict(self,x):
        self.eval()
        return self.forward(x)

    def get_attention(self, x):
        print ('input{}'.format(x.size()))
        output={}
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6,x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        output.setdefault('64_level_1',x1)
        output.setdefault('difference_among2_level',x-x1)
        x = self.outc(x)
        output.setdefault('score',x)
        return output

    def get_net_name(self):
        return 'deep unet'

    def adaptive_bn(self,if_enable=False):
        if if_enable:
            for name,module  in self.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        #if 'down' in name or 'up' in name or 'inc' in name:
                            module.train()
                            module.track_running_stats=True
    def init_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d):
                print (module)

                #if 'down' in name or 'up' in name or 'inc' in name:
                #module.reset_parameters()
                    # print(name, module)
                module.running_mean.zero_()
                module.running_var.fill_(1)

if __name__ == '__main__':
    model = UNet(n_channels=1,n_classes=4)
    model.eval()
    print (list(model.named_children()))
    image = torch.autograd.Variable(torch.randn(1, 1, 512, 512), volatile=True)
    print (model(image)[0].size())
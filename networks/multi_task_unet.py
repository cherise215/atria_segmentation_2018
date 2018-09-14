#!/usr/bin/python
# full assembly of the sub-parts to form the complete net

# python 3 confusing imports :(
from networks.unet_parts import *
from networks.utils import  *


class MT_Net(nn.Module):
    def __init__(self, n_channels, n_classes, n_labels=2, spp_grid=(4, 2, 1),
                 if_dropout=False, upsample_type='bilinear', SegDropout=False):
        super(MT_Net, self).__init__()
        self.if_dropout = if_dropout
        self.output_num = spp_grid
        self.inc = inconv(n_channels, 64)
        self.output_num = spp_grid
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.down5 = down(512,512)
        self.up1 = up(512,512,512,type=upsample_type)
        self.up2 = up(512,512,256,type=upsample_type)
        self.up3 = up(256,256, 128,type=upsample_type)
        self.up4 = up(128,128, 64,type=upsample_type)
        self.up5 = up(64,64, 64,type=upsample_type)
        self.outc = outconv(64, n_classes)

        dim=0
        for a in self.output_num:
            dim += a**2
        print ('dim,',dim)
        self.segdrop=SegDropout
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(dim*512, 512)
        self.fc2 = nn.Linear(512, n_labels)


    def predict(self,x):
        self.eval()
        x1 = self.inc(x)   #N*64*H*w
        x2 = self.down1(x1) #N*128*h/2*w/2
        x3 = self.down2(x2)  #N*256*h/4*w/4
        x4 = self.down3(x3)   #N*512*h/8*w/8
        x5 = self.down4(x4)   #N*512*h/16*w/16
        x6 = self.down5(x5)   #N*512*h/32*w/32
        m1 = self.up1(x6,x5)   #N*512*h/16*w/16
        m2 = self.up2(m1, x4)   #N*256*h/8*w/8
        m3 = self.up3(m2, x3)   #N*128*h/4*w/4
        m4 = self.up4(m3, x2)   #N*64*h/2*w/2
        m5 = self.up5(m4, x1)   #N*64*h/1*w/1
        x = self.outc(m5)
        return x



    def forward(self, x):

        x1 = self.inc(x)   #N*64*H*w
        x2 = self.down1(x1) #N*128*h/2*w/2
        x3 = self.down2(x2)  #N*256*h/4*w/4
        x4 = self.down3(x3)   #N*512*h/8*w/8
        x5 = self.down4(x4)   #N*512*h/16*w/16
        x6 = self.down5(x5)   #N*512*h/32*w/32
        m1 = self.up1(x6,x5)   #N*512*h/16*w/16
        m2 = self.up2(m1, x4)   #N*256*h/8*w/8
        m3 = self.up3(m2, x3)   #N*128*h/4*w/4
        m4 = self.up4(m3, x2)   #N*64*h/2*w/2
        m5 = self.up5(m4, x1)   #N*64*h/1*w/1
        if self.segdrop:
            m5=F.dropout(m5,p=0.1)
        x = self.outc(m5)
        #### classifier
        output=[]
        output.append(x)
        spp = spatial_pyramid_pool(x5, x5.size(0), [int(x5.size(2)), int(x5.size(3))], out_bin_sizes=self.output_num)
        print(spp.size)
        fc1 = self.fc1(spp)
        if self.if_dropout:
            fc1 = self.dropout1(fc1)
        fc2 = self.fc2(fc1)
        class_output=fc2
        output.append(class_output)
        return output

    def get_net_name(self):
        return 'multi task network'

    def fixed_bn(self):
        for name, module in self.named_modules():
           # print(name, module)
            if isinstance(module, nn.BatchNorm2d):
              # print(module[0])
               module.eval()


if __name__=='__main__':

    model = MT_Net(n_channels=1, n_classes=2, spp_grid=[8, 4, 1])
    init_weights(model,'xavier')
    model.eval()
    image = torch.autograd.Variable(torch.rand(8, 1, 256, 256))
    with torch.no_grad():
        segoutput=model(image)
    print(model.predict(image).size())

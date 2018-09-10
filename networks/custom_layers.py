import torch
import torch. nn as nn
import torch.nn.functional as F
def bilinear_additive_upsampling(x, output_channel_num ):
    """
    pytorch implementation of Bilinear Additive Upsampling
    ref: @MikiBear_
    Tensorflow Implementation of Bilinear Additive Upsampling.
    Reference : https://arxiv.org/abs/1707.05847
    https://gist.github.com/mikigom/bad72795c5e87e3caa9464e64952b524
    """

    input_channel=x.size(1)
    assert input_channel>output_channel_num
    assert input_channel % output_channel_num == 0, 'input channel must could be equally divided by output_channel_num '
    channel_split = int(input_channel / output_channel_num)

    print (channel_split)

    new_h= x.size(2)*2
    new_w=x.size(3)*2
    upsampled_op=torch.nn.Upsample(scale_factor=2,mode='bilinear')
    upsampled_x=upsampled_op(x)

    print (upsampled_x.size())

    result=torch.zeros(x.size(0),output_channel_num,new_h,new_w)
    for i in range(0,output_channel_num):
        splited_upsampled_x = upsampled_x.narrow(1,start=i * channel_split,length=channel_split)
        result[:,i,:,:]=torch.sum(splited_upsampled_x,1)

    ## by default, should be cuda tensor.
    result=result.cuda()
    return result



##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'



##################################################################################
# Normalization layers
##################################################################################
class AdaptiveBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply batchNorm
        #x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        #nn.BatchNorm2d
        out = F.batch_norm(
            x, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'




if __name__ == '__main__':
    x = torch.randn(20, 64, 100, 100)
    y = bilinear_additive_upsampling(x, 4)
    print (y.size())
    F.batch_norm()
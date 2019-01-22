import torch
import torch.nn as nn
from torch.optim import lr_scheduler


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


import math
def spatial_pyramid_pool(previous_conv, batch_size, previous_conv_size, out_bin_sizes):
    '''
    ref: Spatial Pyramid Pooling in Deep ConvolutionalNetworks for Visual Recognition
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(0, len(out_bin_sizes)):
        print(previous_conv_size)
        #assert  previous_conv_size[0] % out_bin_sizes[i]==0, 'please make sure feature size can be devided by bins'
        h_wid = int(math.ceil(previous_conv_size[0] / out_bin_sizes[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_bin_sizes[i]))
        # h_stride = int(math.floor(previous_conv_size[0] / out_bin_sizes[i]))
        # w_stride = int(math.floor(previous_conv_size[1] / out_bin_sizes[i]))
        h_pad = (h_wid * out_bin_sizes[i] - previous_conv_size[0] + 1) // 2
        w_pad = (w_wid * out_bin_sizes[i] - previous_conv_size[1] + 1) // 2
        maxpool = nn.MaxPool2d(kernel_size=(h_wid, w_wid), stride=(h_wid, w_wid),padding=(h_pad,w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(batch_size, -1)
            #print("spp size:",spp.size())
        else:
           # print("size:",spp.size())
            spp = torch.cat((spp, x.view(batch_size, -1)), dim=1)
   # print("spp size:",spp.size())

    return spp


'''
https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/4
'''
class GradientReversalFunction(torch.autograd.Function):
    def __init__(self, Lambda):
        super(GradientReversalFunction, self).__init__()
        self.Lambda = Lambda
    def forward(self, input):
        return input.view_as(input)

    def backward(self, grad_output):
        # Multiply gradient by -self.Lambda
        return self.Lambda * grad_output.neg()

class GradientReversalLayer(nn.Module):
    def __init__(self, Lambda, use_cuda=False):
        super(GradientReversalLayer, self).__init__()
        self.Lambda = Lambda
        if use_cuda:
            self.cuda()

    def forward(self, input):
        return GradientReversalFunction(self.Lambda)(input)

    def change_lambda(self, Lambda):
        self.Lambda = Lambda


def gram_matrix_2D(y):
    '''
    give torch 4d tensor, calculate Gram Matrix
    :param y:
    :return:
    '''
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to a fixed number"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_scheduler(optimizer, lr_policy,lr_decay_iters=5,epoch_count=None,niter=None,niter_decay=None):
    print('lr_policy = [{}]'.format(lr_policy))
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + epoch_count - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.5)
    elif lr_policy == 'step2':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)
    elif lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler

def cal_cls_acc(pred,gt):
    '''
    input tensor
    :param pred: network output N*n_classes
    :param gt: ground_truth N [labels_id]
    :return: float acc
    '''
    pred_class = pred.data.max(1)[1].cpu()
    sum = gt.cpu().eq(pred_class).sum()
    count = gt.size(0)
    return sum, count

import collections
import numbers
import random

import numpy as np
import torch
from scipy.misc import imresize
from skimage import transform as sktform


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        '''

        :param img: C*H*W, C>=1
        :param mask: H*W
        :return:
        '''
        mask=mask*1.0
        img=img*1.0
        for a in self.augmentations:
            img, mask = a(img, mask)
        return np.array(img), np.array(mask, dtype=np.uint8)

def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret

class AddGaussianNoise(object):
    """Add gaussian noise to a numpy.ndarray (C xH x W )
    """

    def __init__(self, mean, sigma, random_state=np.random):
        self.sigma = sigma
        self.mean = mean
        self.random_state = random_state

    def __call__(self, image,mask):
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma, random_state=self.random_state)
        else:
            sigma = self.sigma
        if isinstance(self.mean, collections.Sequence):
            mean = random_num_generator(self.mean, random_state=self.random_state)
        else:
            mean = self.mean
        ch, row, col = image.shape
        gauss = self.random_state.normal(mean, sigma, (ch,row, col))
        gauss = gauss.reshape(ch,row, col)
        image += gauss
        return image,mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):

        if random.random() < 0.5:
            return img[:,:,::-1], mask[:,::-1]
        return img, mask


class RandomVerticalFlip(object):
    def __call__(self, img,mask):
        if random.random() < 0.5:

            return img[:,::-1,:], mask[::-1,:]
        return img, mask

class RandomCropNumpy(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, random_state=np.random):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.random_state = random_state

    def __call__(self, img,mask):
        w, h = img.shape[1],img.shape[2]
        th, tw = self.size
        if w == tw and h == th:
            return img
        x1 = self.random_state.randint(0, w - tw)
        y1 = self.random_state.randint(0, h - th)
        return img[:,x1:x1 + tw, y1: y1 + th],mask[x1:x1 + tw, y1: y1 + th]




from torchsample.transforms import RandomAffine
class Affine(object):
    def __call__(self,image,label=None):
        tform = RandomAffine(rotation_range=10, translation_range=[0.1,0.1], zoom_range=(0.7, 1.3),interp=['bilinear','nearest'])
        img = image
        img = torch.from_numpy(img.copy()).float()
        mask = np.expand_dims(label, 0)
        mask = torch.from_numpy(mask.copy()).long()
        t1_slice_tform, mask_slice_tform = tform(img, mask)
        img= t1_slice_tform.numpy()
        mask=mask_slice_tform[0].numpy()
        image=img
        mask=mask.astype(int)
        return image,mask

class RandomScale(object):
    def _init__(self, scale,type=None,mode='symmetric'):
        '''
        random scale 2D image with 2D label
        :param scale:
        :param type:
        :param mode:
        :return:
        '''
        self.scalor=ScaleImage(scale,type,mode)
    def __call__(self,img,mask):
        if random.random() < 0.5:
            return self.scalor(img,mask)
        return img, mask


class RandomGammaCorrection(object):

    def __call__(self,img,mask):
        '''
        support n-d data
        :param img:
        :param mask:
        :return:
        '''
        if random.random() < 0.5:
            gamma = random.random() * 1.2 +0.8  #  0.8-2.0
            # print ('gamma: %f',gamma)
            img = img ** (1 / gamma)
            return img,mask
        return img, mask



class Resize(object):
    def __init__(self, newh,neww,intep):
        self.h = newh
        self.w = neww
        self.intep=intep  #'bilinear' 'nearest']
        assert len(self.intep)==2

    def __call__(self, img, mask):
        if len(img.shape)==2 and len(mask.shape)==2:
            out_image = imresize(img[:,:], (self.h, self.w), interp=self.intep[0])
            out_mask = imresize(mask, (self.h, self.w), interp=self.intep[1])
        if len(img.shape)==3 and len(mask.shape)==2:
            out_image = imresize(img[0, :, :], (self.h, self.w), interp=self.intep[0])
            out_mask = imresize(mask, (self.h, self.w), interp=self.intep[1])
            out_image=out_image.reshape((1,out_image.shape[0],out_image.shape[2]))

        return out_image,out_mask


class ScaleImage(object):
    """
    resize  a 2D numpy array using skimage , support float type
    ref:http://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
    """

    def _init__(self, scale,type=None,mode='symmetric'):
        self.scale = scale  # scale must be tuple indicating scales along different sides
        self.mode = mode
        if type is None:
           self. type=['image','label']
        else:
           self.type=type
        self.order_dict={'image':3,'label':0}

    def __call__(self, input=None, target=None):
        if  input:
            input=sktform.transform.rescale(input, scale=self.scale, order=self.order_dict[self.type[0]], mode=self.mode, cval=0, clip=True, preserve_range=True)
        if target:
            target=sktform.transform.rescale(input, scale=self.scale, order=self.order_dict[self.type[1]], mode=self.mode, cval=0, clip=True, preserve_range=True)

        return input,target


class ScalePower2(object):
    def __init__(self,base=16):#
        self.base=base
    def __call__(self,input, target):
        '''
        make the edge of it  can be divided by the base
        :param input:  C=1, C*H*W image input
        :param target: H*W target label
        :return: inpout
        '''
        assert input.shapep[0]==1 and len(input.shape)==3
        oh, ow = input.shape[1],input.shape[2]
        h = int(round(oh / self.base) * self.base)
        w = int(round(ow / self.base) * self.base)
        if (h == oh) and (w == ow):
            return input,target
        else:
            resizer=Resize(newh=oh,neww=ow,intep=['bilinear' 'nearest'])
            resized_input,resized_target=resizer(input,target)
        return resized_input,resized_target

class RandomCrop(object):
    def __init__(self,new_h,new_w):
        self.new_h=new_h
        self.new_w=new_w
       
    def __call__(self,input,target):
        orig_h = input.shape[0]
        orig_w = input.shape[1]
        shift_h = orig_h - self.new_h
        shift_w = orig_w - self.new_w
        assert shift_h >= 0 and shift_w >= 0
        random_shift_h = np.random.randint(0, shift_h)
        random_shift_w = np.random.randint(0, shift_w)

        cropped_input=input[random_shift_h:random_shift_h + orig_h, 
                      random_shift_w:random_shift_w + orig_w]
        cropped_target = input[random_shift_h:random_shift_h + orig_h,
                        random_shift_w:random_shift_w + orig_w]
        return cropped_input,cropped_target


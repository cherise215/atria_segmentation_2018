

import numpy as np


class CropPad(object):
    def __init__(self,h,w,chw=False):
        '''
        if image > taget image size, simply cropped
        otherwise, pad image to target size.
        :param h: target image height
        :param w: target image width
        '''
        self.target_h = h
        self.target_w = w
        self.chw=chw

    def __call__(self,img):
        # center padding/cropping
        if len(img.shape)==3:
            if self.chw:
                x,y=img.shape[1],img.shape[2]
            else:
                x,y=img.shape[0],img.shape[1]
        else:
            x, y = img.shape[0], img.shape[1]

        x_s = (x - self.target_h) // 2
        y_s = (y - self.target_w) // 2
        x_c = (self.target_h - x) // 2
        y_c = (self.target_w - y) // 2
        if len(img.shape)==2:

            if x>self.target_h and y>self.target_w :
                slice_cropped = img[x_s:x_s + self.target_h , y_s:y_s + self.target_w]
            else:
                slice_cropped = np.zeros((self.target_h, self.target_w), dtype=img.dtype)
                if x<=self.target_h and y>self.target_w:
                    slice_cropped[x_c:x_c + x, :] = img[:, y_s:y_s + self.target_w]
                elif x>self.target_h>0 and y<=self.target_w:
                    slice_cropped[:, y_c:y_c + y] = img[x_s:x_s + self.target_h, :]
                else:
                    slice_cropped[x_c:x_c + x, y_c:y_c + y] = img[:, :]
        if len(img.shape)==3:
            if not self.chw:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[x_s:x_s + self.target_h, y_s:y_s + self.target_w, :]
                else:
                    slice_cropped = np.zeros((self.target_h, self.target_w, img.shape[2]), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[x_c:x_c + x, :, :] = img[:, y_s:y_s + self.target_w, :]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:, y_c:y_c + y, :] = img[x_s:x_s + self.target_h, :, :]
                    else:
                        slice_cropped[x_c:x_c + x, y_c:y_c + y, :] = img
            else:
                if x > self.target_h and y > self.target_w:
                    slice_cropped = img[:,x_s:x_s + self.target_h, y_s:y_s + self.target_w]
                else:
                    slice_cropped = np.zeros((img.shape[0],self.target_h, self.target_w), dtype=img.dtype)
                    if x <= self.target_h and y > self.target_w:
                        slice_cropped[:,x_c:x_c + x, :] = img[:,:, y_s:y_s + self.target_w]
                    elif x > self.target_h > 0 and y <= self.target_w:
                        slice_cropped[:,:, y_c:y_c + y] = img[:,x_s:x_s + self.target_h, :]
                    else:
                        slice_cropped[:,x_c:x_c + x, y_c:y_c + y] = img


        return slice_cropped


    def __repr__(self):
        return self.__class__.__name__ + 'padding to ({0}, {1})'. \
            format(self.target_h, self.target_w)



class ReverseCropPad(object):
    def __init__(self,h,w):
        '''
        :param h: original image height
        :param w: original image width
        '''
        self.h = h
        self.w = w

    def __call__(self,slices_cropped):
        if len(slices_cropped.shape)==2:
            # input N*H*W
            # center padding/cropping
            target_h, target_w = slices_cropped.shape[0], slices_cropped.shape[1]
            result_stack = np.zeros(( self.h, self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[ x_s:x_s + target_h, y_s:y_s + target_w] = slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:, y_s:y_s + target_w] = slices_cropped[x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[x_s:x_s + target_h, :] = slices_cropped[ :, y_c:y_c + self.w]
                else:
                    result_stack = slices_cropped[ x_c:x_c + self.h, y_c:y_c + self.w]

        elif len(slices_cropped.shape)==3:
            # input N*H*W
            # center padding/cropping
            target_h,target_w = slices_cropped.shape[1],slices_cropped.shape[2]
            result_stack=np.zeros((slices_cropped.shape[0],self.h,self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[:,x_s:x_s + target_h , y_s:y_s + target_w]=slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:,:, y_s:y_s + target_w]=slices_cropped[:,x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[:,x_s:x_s + target_h, :]=slices_cropped[:, :,y_c:y_c + self.w]
                else:
                    result_stack=slices_cropped[:,x_c:x_c + self.h, y_c:y_c + self.w]
        elif len(slices_cropped.shape) == 4:
            # input N*C*H*W
            # center padding/cropping
            target_h, target_w = slices_cropped.shape[2], slices_cropped.shape[3]
            result_stack = np.zeros((slices_cropped.shape[0], slices_cropped.shape[1],self.h, self.w))
            x_s = (self.h - target_h) // 2
            y_s = (self.w - target_w) // 2
            x_c = (target_h - self.h) // 2
            y_c = (target_w - self.w) // 2

            if self.h > target_h and self.w > target_w:
                result_stack[:, :,x_s:x_s + target_h, y_s:y_s + target_w] = slices_cropped
            else:
                if self.h <= target_h and self.w > target_w:
                    result_stack[:, :,:, y_s:y_s + target_w] = slices_cropped[:,:, x_c:x_c + self.h, :]
                elif self.h > target_h and self.w <= target_w:
                    result_stack[:,:, x_s:x_s + target_h, :] = slices_cropped[:,:, :, y_c:y_c + self.w]
                else:
                    result_stack = slices_cropped[:, :,x_c:x_c + self.h, y_c:y_c + self.w]

        return result_stack

    def __repr__(self):
        return self.__class__.__name__ + 'recover to ({0}, {1})'. \
            format(self.h, self.w)



class NormalizeMedic(object):
    """
    Normalises given slice/volume to zero mean
    and unit standard deviation.
    """

    def __init__(self,
                 norm_flag=True):
        """
        :param norm_flag: [bool] list of flags for normalisation
        """
        self.norm_flag = norm_flag

    def __call__(self, *inputs):
        # prepare the normalisation flag
        if isinstance(self.norm_flag, bool):
            norm_flag = [self.norm_flag] * len(inputs)
        else:
            norm_flag = self.norm_flag

        outputs = []
        for idx, _input in enumerate(inputs):
            if norm_flag[idx]:
                # subtract the mean intensity value
                mean_val = np.mean(_input.numpy().flatten())
                _input = _input.add(-1.0 * mean_val)

                # scale the intensity values to be unit norm
                std_val = np.std(_input.numpy().flatten())
                _input = _input.div(float(std_val))

            outputs.append(_input)

        return outputs if idx >= 1 else outputs[0]


class ResampleImage(object):
    """
    resampling 3d image volume
    """
    def __init__(self, target_spacing=None,interpolator=None):
        self.target_spacing = target_spacing
        self.interpolator = interpolator
    def __call__(self, image,origin_space=None):
        assert len(image.shape)==3




if __name__ == '__main__':
    image= np.arange(0,15)
    image=np.tile(image,11)
    image=image.reshape((11,15))
    print (image)
    cropped=CropPad(5,19)

    result=cropped(image)
    print (result.shape)
    print (result)

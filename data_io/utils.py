import numpy as np


def merge_mip_stack(whole_image,index,slice,portion=4):
    '''
    select a blob (depth=1/portion) image stacks contains its slice in the center to make mip slices via maximum intensity projection and min intensity projection.
    :param whole_image: n*h*w
    :param index:
    :return:
    '''
    original_data_depth = whole_image.shape[0]
    h,w=whole_image.shape[1],whole_image.shape[2]
    select_depth = int(original_data_depth / portion)
    id_slice=index
    if id_slice < original_data_depth - select_depth // 2 and id_slice >= select_depth // 2:
        blob_image = whole_image[(id_slice - select_depth // 2):(id_slice + select_depth // 2)]
    elif id_slice < select_depth // 2:
        blob_image = whole_image[0:select_depth]
    else:
        blob_image = whole_image[original_data_depth - select_depth:]
    assert blob_image.shape[0] == select_depth
    min_projection = MIP(blob_image, method='min')
    max_projection = MIP(blob_image, method='max')
    temp_input = np.zeros((3, h, w))
    assert len(slice.shape)==2,"input slice must be 2d slice"
    temp_input[0] = slice
    temp_input[1] = min_projection
    temp_input[2] = max_projection
    return temp_input

def automatic_gamma_correction(input_arr):
    '''
    implement automatic gamma correction using
    Somasundaram, K., & Kalavathi, P. (2011).
    Medical image contrast enhancement based on gamma correction.
    Int J Knowl Manag e-learning, 3(1), 15-18.
    '''
    shape=input_arr.shape
    min_val, max_val = np.percentile(input_arr, (5,95))
    mean=np.percentile(input_arr,50)
    if (mean - max_val)==0:
        return input_arr
    else:
        g = np.log(( mean-min_val) / 1.0*( max_val-mean))
        g = np.clip(g,0.8,1.2)
        vec=input_arr.flatten()
        vec=[ i**(1/g)for i in vec]
        return np.array(vec).reshape(shape)


def MIP (blob_image,method='max'):
    '''
    input N*H*W
    return H*W
    '''
    assert isinstance(blob_image, (list, tuple, np.ndarray))
    if method=='max':
        best = np.argmax(blob_image, 0)
    elif method== 'min':
        best = np.argmin(blob_image,0)
    else:
        return NotImplementedError
    length=blob_image.shape[0]
    h,w=blob_image.shape[1],blob_image.shape[2]
    blob_image = blob_image.reshape((length,-1)) # image is now (blob_slice_number, nr_pixels)
    blob_image = blob_image.transpose() # image is now (nr_pixels, stack)
    rebuild_2 = blob_image[np.arange(len(blob_image)), best.ravel()] # Select the right pixel at each location
    rebuild_2 = rebuild_2.reshape((h,w))
    return rebuild_2
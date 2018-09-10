import os
import argparse
from nipy.labs.mask import largest_cc
import torch
import SimpleITK as sitk
from torch.autograd import Variable
from skimage.exposure import equalize_adapthist
from time import time

from networks.multi_task_unet import MT_Net
from data_io.utils import automatic_gamma_correction,merge_mip_stack
from data_io.my_transformer import *
from common.file_io import save_nrrd_result, load_nrrd

N_CLASEES = 2
SPP_GRID = [8, 4, 1]
PREDICT_IMAGE_NAME = 'lgemri.nrrd'


def preprocess_image(if_mip,if_gamma, if_clahe,full_stack_image,force_norm=False):
    '''
     preprocess data for prediction

    :param if_mip: boolean: if use MIP as preprocess
    :param if_gamma: boolean: if use automatic gamma  as pre-process
    :param if_clahe: boolean: if use clahe as pre-process
    :param full_stack_image: a total number of slices from one patient nrrd file:  N*H*W
    :param force_norm: boolean: if force norm
    :return: result_images: ndarray data N*C*H*W,C=3 if use MIP info otherwise set C=1
             origin_h: original image height for future recover
             origin_w:original image width for future recover

    '''
    origin_h,origin_w= full_stack_image.shape[1],full_stack_image.shape[2]
    new_h =origin_h if origin_h%32==0 else 32 * (np.int(origin_h/ 32) + 1)
    new_w = origin_w if origin_w%32==0 else 32 * (np.int(origin_w/ 32) + 1)
    print('new size: {} ,{}'.format(str(new_h),str(new_w)))
    #new_h,new_w=576,576
    CP=CropPad(new_h,new_w,chw=True)
    if not if_mip:
        result_images= np.zeros((full_stack_image.shape[0],1,new_h,new_w),dtype=np.float32) #N*1*H*W
    else:
        result_images= np.zeros((full_stack_image.shape[0],3,new_h,new_w),dtype=np.float32) #N*1*H*W
    full_stack_image=full_stack_image
    for i in range(result_images.shape[0]):

        if if_clahe:
            temp=equalize_adapthist(full_stack_image[i,:,:])
        if if_gamma:
            temp=automatic_gamma_correction(full_stack_image[i,:,:])
        if if_mip:
            new_stack=merge_mip_stack(full_stack_image,i,temp,portion=4) #3*H*W
           # print('bef,',new_stack.shape)
            temp = CP(new_stack)  # pad or crop image to desired size without any rescale/resampling
        else:
            temp=full_stack_image[i,:,:]
           # print('after,',temp.shape)
        if len(temp.shape)==2:
            temp = CP(temp)  # pad or crop image to desired size without any rescale/resampling
            temp = np.expand_dims(temp, 0)
        if (not if_clahe) or force_norm:
            # normalize data
            temp = temp * 1.0
            new_input_mean = np.mean(temp, axis=(1, 2), keepdims=True)
            temp -= new_input_mean
            new_std = np.std(temp, axis=(1, 2), keepdims=True)
            temp /= new_std + 0.00000000001
        result_images[i,:,:,:]=temp
    return result_images,origin_h,origin_w

def preprocess_sequence_image(sequence_length,if_gamma, if_clahe,full_stack_image,force_norm=False):
    '''

     preprocess data to generate a stack of  [i-k...i,i+k] for each slice i
    :param sequence_length: int: how many consequent images to be stacked
    :param if_gamma: boolean: whether use gamma correction during data preprocessing
    :param if_clahe: boolean: whether use CLAHE during data preprocessing
    :param full_stack_image: N*H*W
    :param force_norm: booleanL whether do std normalization
    :return: result_images:N*sequence_length*H_new*W_new
             origin_h: original image height for future recover
             origin_w:original image width for future recover

    '''
     # h x w x n_slices
     # pad
    origin_h,origin_w= full_stack_image.shape[1],full_stack_image.shape[2]
    new_h =origin_h if origin_h%32==0 else 32 * (int(origin_h/ 32) + 1)
    new_w = origin_w if origin_w%32==0 else 32 * (int(origin_w/ 32) + 1)
    print('new_size: {} ,{}'.format(str(new_h),str(new_w)))
    CP=CropPad(new_h,new_w,chw=True)

    result_images= np.zeros((full_stack_image.shape[0],sequence_length,new_h,new_w),dtype=np.float32) #N*1*H*W


    full_stack_image=full_stack_image
    for i in range(4,result_images.shape[0]-sequence_length//2):
        blob_image=full_stack_image[i-sequence_length//2:i+sequence_length//2+1]
        temp=np.zeros(shape=blob_image.shape,dtype=np.float32)
        for j in range(blob_image.shape[0]):
            if if_clahe:
                temp[j]=equalize_adapthist(blob_image[j,:,:])
            if if_gamma:
                temp[j]=automatic_gamma_correction(blob_image[j,:,:])
        temp = CP(temp)
        if (not if_clahe) or force_norm:
            # normalize data
            temp = temp * 1.0
            new_input_mean = np.mean(temp, axis=(1, 2), keepdims=True)
            temp -= new_input_mean
            new_std = np.std(temp, axis=(1, 2), keepdims=True)
            temp /= new_std + 0.00000000001
        result_images[i,:,:,:]=temp
    return result_images,origin_h,origin_w

def predict_img(sequence,if_mip,if_gamma,if_clahe, n_classes,full_img, batch_size=4,if_force_norm=False,sequence_length=1,post_process=False):
    '''
     predict a whole 3D image data of a patient
    :param sequence: Boolean: if the network is desined to process with sequence images
    :param if_mip: Boolean: if use MIP as preprocess
    :param if_gamma:  Boolean: if use automatic gamma augmentation as preprocess
    :param if_clahe:  Boolean: if use clahe  as preprocess
    :param n_classes: int: num of predicted class
    :param full_img: np array: original data from one patient (N*H*W)
    :param batch_size: int: the size of batch for prediction
    :param if_force_norm: boolean: do std norm after all preprocess
    :param sequence_length: int: the  sequence length if use sequence data
    :param post_process: boolean: if true, then do morphological operations and keep the largest component for final prediction.
    :return:result: N*H_new*W_new
            original_space_result:N*orig_h*orig_w
            transformed_img:
            result_prob_map
    '''
    global net  ## predictor

    ## prepare data
    if sequence:
        transformed_img,origin_h,origin_w = preprocess_sequence_image(sequence_length,if_gamma,if_clahe,full_img,if_force_norm)

    else:
        transformed_img,origin_h,origin_w = preprocess_image(if_mip,if_gamma,if_clahe,full_img,if_force_norm)
    print (transformed_img.shape)
    data_size = transformed_img.shape[0]
    n_batch=int(np.round(data_size/batch_size))
    index_tuple= [i for i in range(data_size)]
    if n_batch*batch_size<data_size:
        n_batch+=1
        for i in range(n_batch*batch_size-data_size):
            index_tuple.append(data_size-1)

    print(index_tuple)

    result=np.zeros((data_size,transformed_img.shape[2],transformed_img.shape[3]),dtype=np.uint8)
    result_prob_map=np.zeros((data_size,n_classes,transformed_img.shape[2],transformed_img.shape[3]),dtype=np.float32)

    original_space_result=np.zeros((data_size,origin_h,origin_w),dtype=np.uint8)
    print ('dsa',n_batch)
    print ('dsa',batch_size)


    RCP=ReverseCropPad(origin_h, origin_w)
    ### prediction:
    for i in range(n_batch):
        batch_data=transformed_img[i*batch_size:(i+1)*batch_size,:,:,:]
        print (batch_data.shape)
        torch_batch_data=torch.from_numpy(batch_data).float()
        print('tor',torch_batch_data.shape)
        if torch.cuda.is_available():
            net.cuda(0)

            images = Variable(torch_batch_data.cuda(0),volatile=True)
        else:

            images = Variable(torch_batch_data,volatile=True)

        ## predict
        if isinstance(net,torch.nn.DataParallel):
            net_name=net.module.get_net_name()
            with torch.no_grad():
                outputs=net.module.predict(images)
        else:
            with torch.no_grad():
                outputs=net.predict(images)

        pred = outputs.data.max(1)[1].cpu().numpy()
        result[i*batch_size:(i+1)*batch_size]=pred
        result_prob_map[i*batch_size:(i+1)*batch_size]=outputs.data.cpu().numpy()
        ## save prediction to its original size
        rescale_result=RCP(pred)
        original_space_result[i*batch_size:(i+1)*batch_size]=rescale_result


    print (result.shape)
    if post_process:
        if np.sum(original_space_result)>0:
            mask = sitk.GetImageFromArray(original_space_result)
            mask = sitk.BinaryDilate(mask, 2)
            mask = sitk.BinaryErode(mask, 2)
            original_space_result = sitk.GetArrayFromImage(mask)
            voted_mask_binary = largest_cc(original_space_result)
            original_space_result = original_space_result * voted_mask_binary


    return result,original_space_result,result_prob_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='/home/model/model_x.pkg',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filedirs', default='/home/data')
    parser.add_argument('--save_name', '-o', metavar='OUTPUT',
                        help='save filenames of predict',default="predict.nrrd")
    parser.add_argument('--enhance', '-e', action='store_true',
                        help="preprocess image use clahe",default=False)
    parser.add_argument('--gamma', '-g', action='store_true',
                        help="preprocess image use gamma correction", default=False)
    parser.add_argument('--force_norm', action='store_true',
                        help="always do normalization ", default=False)
    parser.add_argument('--batch_size',type=int,
                        help="batch size ", default=1)
    parser.add_argument('--post',  action='store_true',
                        help="post-processing ", default=False)
    parser.add_argument('--upsample_type', type=str, dest='upsample_type',
                      help='use  which method for upsampling, bilinear, bilinear_additive or deconv',
                      default='bilinear')

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))


    patient_dir_list = sorted(os.listdir(args.input))
    patient_path_list=[os.path.join(args.input,name) for name in patient_dir_list]

    print("Loading model ...{}".format(args.model))
    cache=torch.load(args.model)

    ## load preprocess data config from saved model
    try:
        if_mip=cache['if_mip']
        if_clahe=cache['if_clahe']
        print ('if_mip: ',if_mip)
    except:
        if_mip=False
        if_clahe=False


    ## if specify config
    if_clahe=True if args.enhance else if_clahe
    print('if_clahe: ', if_clahe)
    sequence_length=1

    try:
        sequence_length=cache['sequence_length']
        print('sequence_length: ', sequence_length)
        if sequence_length>1:
            sequence=True
        else:
            sequence=False
    except:
        sequence=False

    print('if sequence: ', sequence)
    n_channel=1 if not sequence else sequence_length

    print('initialize with mt model')
    net = MT_Net(n_labels=2, n_channels=n_channel, spp_grid=SPP_GRID, n_classes=N_CLASEES, if_dropout=False, upsample_type=args.upsample_type)

    print (net.get_net_name())
    net.load_state_dict(cache['model_state'])
    net = torch.nn.DataParallel(net, device_ids=[0])
    try:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
    except:
        raise NotImplementedError
    print("Model loaded !")

    ##predict
    net.eval()
    t=0
    for i, fn in enumerate(patient_path_list):

        start_time=time()
        print("\nPredicting image {} ...".format(fn))
        path=os.path.join(fn,PREDICT_IMAGE_NAME)
        if not os.path.exists(path):continue
        raw_data,sitk_image=load_nrrd(os.path.join(fn,PREDICT_IMAGE_NAME))
        print (raw_data.shape)
        ##
        if 'coronal' in args.model:
            raw_data= np.transpose(raw_data, (1, 0, 2))
        elif 'sagittal' in args.model:
            raw_data= np.transpose(raw_data, (2, 0, 1))

        result,original_result,result_prob_map= predict_img(sequence=sequence,if_mip=if_mip,if_gamma=args.gamma,
                                                                        if_clahe=if_clahe, n_classes=N_CLASEES,
                                                                        full_img=raw_data, batch_size=args.batch_size,if_force_norm=args.force_norm,sequence_length=sequence_length,post_process=args.post)
        mask_name=args.save_name

        if 'coronal' in args.model:
            original_result = np.transpose(original_result, (1, 0, 2))
        elif 'sagittal' in args.model:
            original_result = np.transpose(original_result, (1, 2, 0))

        t+= time()-start_time
        save_nrrd_result(file_path=os.path.join(fn,mask_name),data=original_result,reference_img=sitk_image)

    print ('average time', t*1.0/len(patient_path_list))


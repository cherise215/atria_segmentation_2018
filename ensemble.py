import os
from nipy.labs.mask import largest_cc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from os.path import join
import SimpleITK as sitk
from time import time

from networks.multi_task_unet import MT_Net
from data_io.my_transformer import *
from common.file_io import save_nrrd_result, load_nrrd
from predict_nrrd import preprocess_image

if __name__=='__main__':


    hard_ensemble=False # use aggregated prob instead of max-voting to get more smoothed result

    model_dict = {
        'multi_task_network_1': 'models/atria_mt_model_dataset_0.pkl',
        'multi_task_network_2': 'models/atria_mt_model_dataset_1.pkl',
        'multi_task_network_3': 'models/atria_mt_model_dataset_2.pkl',
        'multi_task_network_4': 'models/atria_mt_model_dataset_3.pkl',
        'multi_task_network_5': 'models/atria_mt_model_dataset_4.pkl'}

    image_name = 'lgemri.nrrd'
    test_dir = '/vol/medic01/users/cc215/data/AtriaSeg_2018_training/AtriaSeg_2018_testing'

    t=0.
    count=0
    for patient_name in sorted(os.listdir(test_dir)):
        start_time=time()
        print ('predict {}'.format(patient_name))
        patient_img_dir=join(test_dir,patient_name)
        if not os.path.isdir(patient_img_dir):
            continue
        count+=1
        patient_img_path=join(patient_img_dir,image_name)

        data,sitk_image=load_nrrd(patient_img_path,dtype=sitk.sitkUInt8)

        ## load model
        model_list=[]
        n_classes=2
        for k,v in model_dict.items():
            if 'multi_task' in k:
                model = MT_Net(n_channels=1, n_classes=2, n_labels=2, if_dropout=False, spp_grid=[8, 4, 1], upsample_type='bilinear')
            else:
                raise NotImplementedError

            cache=torch.load(v)
            try:
                model.load_state_dict(cache['model_state'])
                model = torch.nn.DataParallel(model, device_ids=[0])
            except:
                model = torch.nn.DataParallel(model, device_ids=[0])
                model.load_state_dict(cache['model_state'])

            model.cuda()
            model.eval()
            model_list.append(model)


        ## get confidence score maps

        transformed_img, origin_h, origin_w = preprocess_image(if_mip=False,if_clahe=False,if_gamma=False,full_stack_image=data,force_norm=True)

        batch_size=1

        n_batch=transformed_img.shape[0]//batch_size if transformed_img.shape[0]%batch_size==0 else transformed_img.shape[0]//batch_size+1
        final_prob_map=np.zeros((transformed_img.shape[0],n_classes,transformed_img.shape[2],transformed_img.shape[3]),dtype=np.float32)
        sum_predict=np.zeros((transformed_img.shape[0],transformed_img.shape[2],transformed_img.shape[3]),dtype=np.int64)
        for i in range(n_batch):
            print ('iter:',str(i))
            batch_data=transformed_img[i:i*batch_size+batch_size]
            torch_batch_data = torch.from_numpy(batch_data).float()
            print('tor', torch_batch_data.shape)
            if torch.cuda.is_available():
                images = Variable(torch_batch_data.cuda(0), volatile=True)
            else:

                images = Variable(torch_batch_data, volatile=True)
            ## multi predict:
            count=0
            for model in model_list:
                count+=1
                if isinstance(model, torch.nn.DataParallel):
                    with torch.no_grad():
                        outputs = model.module.predict(images)
                else:
                    with torch.no_grad():
                        outputs = model.predict(images)

                if not hard_ensemble:
                    probmap = F.softmax(outputs) #N*2*H*W
                    final_prob_map[i:i*batch_size+batch_size]+=probmap
                else:
                    single_pred = np.argmax(outputs, axis=1)
                    sum_predict[i:i * batch_size + batch_size] += single_pred ## add count

        ##average score
        if not hard_ensemble:
            final_prob_map=final_prob_map/len(model_list)
        else:
            sum_predict=(sum_predict*1.0)/(1.0*len(model_list))

        ##vote
        if not hard_ensemble:
            ## get result from essembled prob.
            pred = np.argmax(final_prob_map, axis=1)
            pred = np.uint8(pred)
        else:
            ## majority vote
            sum_predict[sum_predict>=0.5] = 1
            sum_predict[sum_predict < 0.5] = 0
            pred=np.uint8(sum_predict*255)
            print(np.sum(pred))

        mask = sitk.GetImageFromArray(pred)
        mask = sitk.BinaryDilate(mask, 2)
        mask = sitk.BinaryErode(mask, 2)
        pred = sitk.GetArrayFromImage(mask)
        voted_mask_binary = largest_cc(pred)

        pred = pred * voted_mask_binary
        t+=time()-start_time

        save_nrrd_result(file_path=os.path.join(patient_img_dir, 'EMMA_soft.nrrd'), data=pred, reference_img=sitk_image)



    print ('average time:', t/count)

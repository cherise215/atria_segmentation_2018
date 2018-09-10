import os
from os.path import join
import pandas as pd
import torch.utils.data as data
import datetime
from skimage.exposure import equalize_adapthist
import SimpleITK as sitk

from data_io.utils import merge_mip_stack
from common.file_io import load_nrrd
from data_io.my_transformer import CropPad
from data_io.data_augmentation import *
from data_io.utils import automatic_gamma_correction


class AtriaDataset(data.Dataset):
    def __init__(self, root_dir,split, extra_label=False, if_subsequent=False,sequence_length=1,if_mip=False,extra_label_csv_path='',augmentation=False, if_clahe=False,if_gamma_correction=False,preload_data=False,input_h=224,input_w=224,orientation=0):
        super(AtriaDataset, self).__init__()
        dataset_dir=join(root_dir,split)
        self.patient_list=os.listdir(dataset_dir)
        self.patient_path_list=sorted([os.path.join(dataset_dir,pid) for pid in self.patient_list])
        self.data_size=len(self.patient_path_list)
        self.if_clahe=if_clahe
        # report the number of images in the dataset
        print('Number of {0} images: {1} nrrds'.format(split, self.data_size))

        # data augmentation
        self.augmentation = augmentation
        self.input_h = input_h
        self.input_w = input_w
        self.split = split
        self.gamma_correction=if_gamma_correction
        self.extra_label=extra_label

        # data load into the ram memory
        self.preload_data = preload_data
        if self.preload_data:
            print('Preloading the {0} dataset ...'.format(split))
            self.raw_images = [load_nrrd(os.path.join(ii,'lgemri.nrrd'),dtype=sitk.sitkUInt8)[0] for ii in self.patient_path_list]
            self.raw_labels = [load_nrrd(os.path.join(ii,'laendo.nrrd'))[0] for ii in self.patient_path_list]
            print('Loading is done\n')

        ## add csv to get class label post ablation/not
        if self.extra_label:
            assert os.path.exists(extra_label_csv_path)
            print ('loading csv as extra label')
            df=pd.read_csv(extra_label_csv_path,header=0)
            self.df=df
        self.if_mip=if_mip
        if self.if_mip:
            print ('mip enabled')
        if self.gamma_correction:
            print ('automatic gamma correction augmented')
        self.if_subsequent=if_subsequent ## use former and next slices
        self.orientation=orientation
        '''
        0: get axial slices
        1: get coronary slices
        2: get sagittal slices 
        '''
        self.sequence_length = sequence_length


    def get_size(self):
        return len(self.patient_list)

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        if not self.preload_data:
            input, _ = load_nrrd(os.path.join(self.patient_path_list[index],'lgemri.nrrd'))
            target, _ = load_nrrd(os.path.join(self.patient_path_list[index],'laendo.nrrd'))
        else:
            input = np.copy(self.raw_images[index])
            target = np.copy(self.raw_labels[index])

        target=np.uint8(target)
        target[target>=1]=1 # binary
        has_object=1 if np.sum(target)>0 else 0

        ###slicing image at given orientation
        if self.orientation==1:
            #coronal
            input = np.transpose(input, (1, 0, 2))
            target = np.transpose(target, (1, 0, 2))#88*h/88*w
        elif self.orientation==2:
            #sagittal
            input = np.transpose(input, (2, 0, 1))
            target = np.transpose(target, (2, 0, 1))

        if self.orientation==0:
            # pass a random slice for the time being
            id = np.random.randint(0, input.shape[0])
            if self.if_subsequent:
                assert self.sequence_length>=3
                half_length=self.sequence_length//2
                id=np.random.randint(half_length,input.shape[0]-half_length)
        else:
            ### for sagittal and coronal slices need to do balance sampling before choose index
            p_indexes, n_indexes = self.get_p_n_group(target)
            p = np.random.rand()
            if p >= 0.5:
                temp_index = np.random.randint(0, len(p_indexes))
                id = p_indexes[temp_index]
               # print('choose p(has object) indexes')

            else:
                temp_index = np.random.randint(0, len(n_indexes))
                id = n_indexes[temp_index]
                #print('choose n (background) indexes')

        if  not self.if_subsequent:
            input = input[[id], :, :]  # c*h*w
        else:
            start= id-self.sequence_length//2
            end =  id+self.sequence_length//2+1
            print ('{} {}'.format(start,end))
            input=input[start:end,:,:]
        target = target[id, :, :]

        original_data = input.copy()
        temp_input = np.zeros(input.shape,dtype=np.float)
        for i in range(input.shape[0]):
            if self.gamma_correction:
                input[i]=automatic_gamma_correction(input[i])

            if self.if_clahe:
                new_input=equalize_adapthist(input[i])
                temp_input[i]=new_input
        if self.if_clahe:
            input=input*1.0
            input=temp_input

        if self.if_mip:
            assert input.shape[0]==1
            ##select a blob which has 1/4 totalslices to combine a projection
            temp_input= merge_mip_stack(original_data,id,input[0],portion=4) # 3chanel
            input=temp_input




        new_input,new_target = self.pair_transform(input,target,self.input_h,self.input_w)
        #normalize data
        new_input=new_input*1.0
        new_input_mean=np.mean(new_input,axis=(1,2),keepdims=True)
        new_input-=new_input_mean
        new_std=np.std(new_input,axis=(1,2),keepdims=True)
        new_input/=new_std+0.00000000001

        input = torch.from_numpy(new_input).float()
        target = torch.from_numpy(new_target).long()
        if not self.extra_label:
            return {'input': input, 'target':  target}
        else:
            patient_name=self.patient_path_list[index].split('/')[-1]
            post_ablation=self.df[self.df['Patient Code']==patient_name]['post_ablation?'].values[0]
            post_ablation = np.bool( post_ablation)
            class_label=1 if post_ablation is True  else 0
            # print (class_label)
            return {'input': input, 'target':  target,'p_name':patient_name, 'post_ablation':class_label,'has_object':has_object}




    def __len__(self):
        return self.data_size

    def get_p_n_group(self,data):
        result = np.sum(data, axis=(1, 2))
        positive_slice_indexes = np.nonzero(result > 0)[0]  ## find slices containing objects
        negative_slice_indexes = np.nonzero(result == 0)[0]
        return positive_slice_indexes, negative_slice_indexes


    def pair_transform(self,image,label,input_h=256,input_w=256):
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)
        label=label.astype(int)
        #print('fd:',image.shape)
        result_image = np.zeros((image.shape[0], input_h, input_w), dtype=np.float32)

        ##data augmentation
        data_aug = Compose([RandomVerticalFlip(),RandomHorizontallyFlip(), Affine(),RandomGammaCorrection()])
        if self.augmentation:
            image, label = data_aug(image, label)


        CPad = CropPad(input_h, input_w)
        for i  in range(result_image.shape[0]):
            result_image[i]=CPad(image[i])
        result_label = CPad(label)
        return result_image, result_label



if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    import torch
    root_dir = '/vol/medic01/users/cc215/data/AtriaSeg_2018_training/dataset'
    extra_label_path='/vol/medic01/users/cc215/data/AtriaSeg_2018_training/pre post ablation recrods.csv'
    torch.cuda.set_device(0)
    image_size=[480]
    for rnd in range(100):
        size_1=np.random.randint(0,1)
        train_dataset = AtriaDataset(root_dir,extra_label=True,if_subsequent=False,if_mip=False,extra_label_csv_path=extra_label_path,if_clahe=True,augmentation=True, split='train',input_h=512,input_w=image_size[size_1],preload_data=False,orientation=0)
        train_loader = DataLoader(dataset=train_dataset, num_workers=16, batch_size=4, shuffle=True)
        n=0
        for epoch_iter, data in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            print (epoch_iter)
            print (data['input'].shape) #10*224*224*1
            print (data['target'].shape)
            print (data['post_ablation'].shape)
            positive_sample=np.count_nonzero(data['post_ablation'].numpy())
            print ('positive_sample',positive_sample)
            print ('ratio',positive_sample/4.0)

            #print ('%s %d' %(data['p_name'] ,data['post_ablation'].numpy()))
            if n==0:
                plt.subplot(141)
                plt.imshow(data['input'][0, 0, :, :])
                plt.subplot(142)
                plt.imshow(data['input'][1,0,:,:])
                plt.subplot(143)
                plt.imshow(data['target'][0,:,:])
                plt.subplot(144)
                plt.imshow((data['target'][1].numpy()))
                plt.show()#
            n+=1

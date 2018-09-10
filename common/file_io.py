import os

import SimpleITK as sitk
def save_nrrd_result(file_path,data,reference_img):
    '''
    save data to a nrrd file
    :param file_path: full path
    :param data: np array
    :param reference_img: refence image
    :return: None
    '''
    image=sitk.GetImageFromArray(data)
    image.SetDirection(reference_img.GetDirection())
    image.SetOrigin(reference_img.GetOrigin())
    image.SetSpacing(reference_img.GetSpacing())
    sitk.WriteImage(image,file_path)


def load_nrrd(full_path_filename,dtype=sitk.sitkUInt8):
    '''
    N*h*W
    :param full_path_filename:
    :return:*H*W
    '''
    if not os.path.exists(full_path_filename):
        raise FileNotFoundError
    image = sitk.ReadImage( full_path_filename )
    image= sitk.Cast( sitk.RescaleIntensity(image),dtype)
    data = sitk.GetArrayFromImage(image) # N*H*W
    return data,image
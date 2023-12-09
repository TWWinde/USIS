import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image
import pydicom
#(320, 60, 320) # 58760


def get_2d_mr_images(image_path):
    n = 58760
    print('There are %d images', len(image_path))
    for i in range(len(image_path)):
        image = pydicom.dcmread(image_path[i])
        image = image.pixel_array
        image = (((image - image.min()) / (image.max() - image.min())) * 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.convert('RGB')
        image = image.rotate(-90, expand=True)
        image.save(f'/misc/data/private/autoPET/ct_mr/mr/slice_{n}.png')
        n += 1

    print(' finished')


def list_images(data_path):

    image_path = []
    train_root_path = os.path.join(data_path, 'CHAOS_Train_Sets', 'MR')
    train_names = os.listdir(train_root_path)
    test_root_path = os.path.join(data_path, 'CHAOS_Test_Sets', 'MR')
    test_names = os.listdir(test_root_path)

    for name in train_names:
        mr_path = os.path.join(train_root_path, name, 'T1DUAL', 'DICOM_anon', 'InPhase')
        mr_list = os.listdir(mr_path)
        for img_name in mr_list:
            if img_name.endswith('.dcm'):
                image_path.append(os.path.join(mr_path, img_name))

    for name in test_names:
        mr_path = os.path.join(test_root_path, name, 'T1DUAL', 'DICOM_anon', 'InPhase')
        mr_list = os.listdir(mr_path)
        for img_name in mr_list:
            if img_name.endswith('.dcm'):
                image_path.append(os.path.join(mr_path, img_name))


    return image_path


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/image', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/mr', exist_ok=True)

    data_path = '/misc/data/private/autoPET/'
    images_path = list_images(data_path)
    get_2d_mr_images(images_path)


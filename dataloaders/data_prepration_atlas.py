import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image
import pydicom
#(320, 60, 320) # 58760


def get_2d_mr_images(image_path):
    n = 60060
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
    train_names = os.listdir(data_path)

    for name in train_names:
        file_path = os.path.join(data_path, name)
        dcm_names = os.listdir(file_path)
        for img_name in dcm_names:
            if img_name.endswith('.dcm') and img_name.startswith('MR'):
                image_path.append(os.path.join(file_path, img_name))

    return image_path


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/image', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/mr', exist_ok=True)

    data_path = '/misc/data/private/autoPET/atlas'
    images_path = list_images(data_path)
    get_2d_mr_images(images_path)


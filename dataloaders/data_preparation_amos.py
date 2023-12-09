import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image

#(320, 60, 320) #37591


def get_2d_mr_images(image_path):
    n = 37591
    for i in range(len(image_path)):
        nifti_img = nib.load(image_path[i])
        img_3d = nifti_img.get_fdata()

        for z in range(5, img_3d.shape[2] - 5):
            img_slice = img_3d[:, :, z]
            image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = image.convert('RGB')
            image = image.resize((200, 320))
            new_image = Image.new("RGB", (320, 320), color="black")
            x_offset = (320 - image.width) // 2
            y_offset = (320 - image.height) // 2
            new_image.paste(image, (x_offset, y_offset))
            image = new_image.rotate(360, expand=True)
            image.save(f'/misc/data/private/autoPET/ct_mr/mr/slice_{n}.png')
            n += 1
    print(' finished')


def list_images(data_path):

    image_path = []
    train_root_path = os.path.join(data_path, 'imagesTr')
    train_names = os.listdir(train_root_path)
    test_root_path = os.path.join(data_path, 'imagesTs')
    test_names = os.listdir(test_root_path)
    val_root_path = os.path.join(data_path, 'imagesVa')
    val_names = os.listdir(val_root_path)

    for name in train_names:
        if name.endswith('nii.gz') and int(name.split('_')[-1].split('.')[0]) >= 500:
            image_path.append(os.path.join(train_root_path, name))
    for name in test_names:
        if name.endswith('nii.gz') and int(name.split('_')[-1].split('.')[0]) >= 500:
            image_path.append(os.path.join(test_root_path, name))
    for name in val_names:
        if name.endswith('nii.gz') and int(name.split('_')[-1].split('.')[0]) >= 500:
            image_path.append(os.path.join(val_root_path, name))

    return image_path


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/image', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/mr', exist_ok=True)

    data_path = '/misc/data/private/autoPET/amos22'
    images_path = list_images(data_path)
    get_2d_mr_images(images_path)

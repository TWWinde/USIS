import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image


def get_2d_mr_images(pelvis_path, brain_path):
    n = 0
    for i in range(len(pelvis_path)):
        nifti_img = nib.load(pelvis_path[i])
        img_3d = nifti_img.get_fdata()

        for z in range(10, img_3d.shape[2] - 5):
            img_slice = img_3d[:, :, z]
            image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = image.convert('RGB')
            new_image = Image.new("RGB", (470, 470), color="black")
            x_offset = (470 - image.width) // 2
            y_offset = (470 - image.height) // 2
            new_image.paste(image, (x_offset, y_offset))
            image = new_image.rotate(-180, expand=True)
            image.save(f'/misc/data/private/autoPET/ct_mr/mr/slice_{n}.png')
            n += 1
    print('pelvis finished')

    for i in range(len(brain_path)):
        nifti_img = nib.load(brain_path[i])
        img_3d = nifti_img.get_fdata()

        for z in range(45, img_3d.shape[2] - 40):
            img_slice = img_3d[:, :, z]
            image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = image.convert('RGB')
            new_image = Image.new("RGB", (470, 470), color="black")
            x_offset = (470 - image.width) // 2
            y_offset = (470 - image.height) // 2
            new_image.paste(image, (x_offset, y_offset))
            image = new_image.rotate(-180, expand=True)
            image.save(f'/misc/data/private/autoPET/ct_mr/mr/slice_{n}.png')
            n += 1
    print("finished brain ")


def list_images(path):
    image_path = []
    names = os.listdir(path)
    for name in names:
        if name != 'overview':
            image_path.append(os.path.join(path, name, 'mr.nii.gz'))

    return image_path


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/image', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/mr', exist_ok=True)

    path_pelvis = "/misc/data/private/autoPET/Task1/pelvis"
    path_brain = "/misc/data/private/autoPET/Task1/brain"

    pelvis_path = list_images(path_pelvis)
    brain_path = list_images(path_brain)
    get_2d_mr_images(pelvis_path, brain_path)


import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image


def get_2d_mr_images(mr_path, ct_path, ct_label_path):
    n = 0
    k = 0
    for i in range(len(mr_path)):
        nifti_mr = nib.load(mr_path[i])
        mr_3d = nifti_mr.get_fdata()
        nifti_ct = nib.load(ct_path[i])
        ct_3d = nifti_ct.get_fdata()
        nifti_ct_label = nib.load(ct_label_path[i])
        ct_label_3d = nifti_ct_label.get_fdata()

        for z in range(5, mr_3d.shape[2] - 5):
            mr_slice = mr_3d[:, :, z]
            ct_slice = ct_3d[:, :, z]
            ct_label_slice = ct_label_3d[:, :, z]
            if ct_label_slice.max() != ct_label_slice.min() and ct_slice.max() != ct_slice.min() and mr_slice.max() != mr_slice.min():
                mr_image = (((mr_slice - mr_slice.min()) / (mr_slice.max() - mr_slice.min())) * 255).astype(np.uint8)
                ct_image = (((ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min())) * 255).astype(np.uint8)
                height, width = mr_image.shape
                target_size_mr = max(height, width)
                height, width = ct_image.shape
                target_size_ct = max(height, width)
                mr_image = Image.fromarray(mr_image)
                mr_image = mr_image.convert('RGB')
                new_image_mr = Image.new("RGB", (target_size_mr, target_size_mr), color="black")

                ct_image = Image.fromarray(ct_image)
                ct_image = ct_image.convert('RGB')
                new_image_ct = Image.new("RGB", (target_size_ct, target_size_ct), color="black")

                x_offset_mr = (target_size_mr - mr_image.width) // 2
                y_offset_mr = (target_size_mr - mr_image.height) // 2
                x_offset_ct = (target_size_mr - mr_image.width) // 2
                y_offset_ct = (target_size_mr - mr_image.height) // 2

                new_image_mr.paste(mr_image, (x_offset_mr, y_offset_mr))
                image_mr = new_image_mr.rotate(-180, expand=True)
                #image_mr.save(f'/misc/data/private/autoPET/CT_MR/mr/slice_{n}.png')

                new_image_ct.paste(ct_image, (x_offset_ct, y_offset_ct))
                image_ct = new_image_ct.rotate(-180, expand=True)

                height, width = ct_label_slice.shape
                target_size = max(height, width)
                square_image = np.zeros((target_size, target_size), dtype=np.uint8)
                x_offset = (target_size - width) // 2
                y_offset = (target_size - height) // 2
                square_image[y_offset:y_offset + height, x_offset:x_offset + width] = ct_label_slice
                rotation_matrix = cv2.getRotationMatrix2D((target_size / 2, target_size / 2), 180, 1)
                rotated_square_image = cv2.warpAffine(square_image, rotation_matrix, (target_size, target_size))

                if k < 500:
                    image_mr.save(f'/misc/data/private/autoPET/CT_MR/mr/val/slice_{k}.png')
                    #image_ct.save(f'/misc/data/private/autoPET/CT_MR/ct/val/images/slice_{k}.png')
                    #cv2.imwrite(f'/misc/data/private/autoPET/CT_MR/ct/val/labels/slice_{k}.png', rotated_square_image)
                else:
                    m = k - 500
                    image_mr.save(f'/misc/data/private/autoPET/CT_MR/mr/train/slice_{m}.png')
                    #image_ct.save(f'/misc/data/private/autoPET/CT_MR/ct/train/images/slice_{m}.png')
                    #cv2.imwrite(f'/misc/data/private/autoPET/CT_MR/ct/train/labels/slice_{m}.png', rotated_square_image)
                n += 1
                k += 1
    print('pelvis finished')


def list_images(path):
    mr_path = []
    ct_path = []
    ct_label_path = []
    names = os.listdir(path)
    for name in names:
        if name != 'overview':
            mr_path.append(os.path.join(path, name, 'mr.nii.gz'))
            ct_path.append(os.path.join(path, name, 'ct.nii.gz'))
            ct_label_path.append(os.path.join('/misc/data/private/autoPET/Task1/ct_label_combine', f'{name}_ct_label.nii.gz'))

    return mr_path, ct_path, ct_label_path


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/CT_MR/ct/train/labels', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/CT_MR/ct/train/images', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/CT_MR/ct/val/images', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/CT_MR/ct/val/labels', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/CT_MR/mr/train', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/CT_MR/mr/val', exist_ok=True)

    path_pelvis = "/misc/data/private/autoPET/Task1/pelvis"
    path_brain = "/misc/data/private/autoPET/Task1/brain"

    mr_path, ct_path, ct_label_path = list_images(path_pelvis)
    get_2d_mr_images(mr_path, ct_path, ct_label_path)

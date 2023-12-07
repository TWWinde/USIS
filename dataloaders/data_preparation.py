import os
import cv2
import nibabel as nib
import numpy as np
from PIL import Image


def ct_array2slices(img_3d, seg_3d, n, mode='train'):
    for z in range(seg_3d.shape[2]):
        seg_slice = seg_3d[:, :, z]
        img_slice = img_3d[:, :, z]
        if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
            for x in range(seg_slice.shape[0]):
                seg_line = seg_slice[x, :]
                if seg_line.max() != seg_line.min():
                    l = x
                    break
            for y in range(seg_slice.shape[1]):
                seg_line = seg_slice[:, y]
                if seg_line.max() != seg_line.min():
                    w = y
                    break
            if l + 480 < seg_3d.shape[0]:
                a = l
                b = l + 480
                if w + 480 >seg_3d.shape[1]:
                    c = seg_3d.shape[1]-480
                    d = seg_3d.shape[1]
                else:
                    c = w
                    d = w + 480
            else:
                a = seg_3d.shape[0]-480
                b = seg_3d.shape[0]
                if w + 480 >seg_3d.shape[1]:
                    c = seg_3d.shape[1]-480
                    d = seg_3d.shape[1]
                else:
                    c = w
                    d = w + 480

            seg_slice = seg_slice[a:b, c:d]
            img_slice = img_slice[a:b, c:d]
            seg_slice = seg_slice.astype(np.uint8)
            # print(img_slice.shape) (240, 120)
            image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = image.convert('RGB')
            image.save(f'/misc/data/private/autoPET/MM-WHS/ct/{mode}/image/slice_{n}.png')
            cv2.imwrite(f'/misc/data/private/autoPET/MM-WHS/ct/{mode}/label/slice_{n}.png', seg_slice)
            n += 1
    return n


def mr_array2slices(img_3d, seg_3d, n, mode='train'):
    for z in range(seg_3d.shape[1]):
        seg_slice = seg_3d[:, z, :]
        img_slice = img_3d[:, z, :]
        if img_slice.max() != img_slice.min() and seg_slice.max() != seg_slice.min():
            for x in range(seg_slice.shape[0]):
                seg_line = seg_slice[x, :]
                if seg_line.max() != seg_line.min():
                    l = x
                    break
            for y in range(seg_slice.shape[1]):
                seg_line = seg_slice[:, y]
                if seg_line.max() != seg_line.min():
                    w = y
                    break
            if l + 120 < seg_3d.shape[0]:
                a = l
                b = l + 120
                if w + 120 >seg_3d.shape[2]:
                    c = seg_3d.shape[2]-120
                    d = seg_3d.shape[2]
                else:
                    c = w
                    d = w + 120
            else:
                a = seg_3d.shape[0]-120
                b = seg_3d.shape[0]
                if w + 120 >seg_3d.shape[2]:
                    c = seg_3d.shape[2]-120
                    d = seg_3d.shape[2]
                else:
                    c = w
                    d = w + 120

            seg_slice = seg_slice[a:b, c:d]
            img_slice = img_slice[a:b, c:d]
            seg_slice = seg_slice.astype(np.uint8)
            # print(img_slice.shape) (240, 120)
            image = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image = image.convert('RGB')
            image.save(f'/misc/data/private/autoPET/MM-WHS/mr/{mode}/image/slice_{n}.png')
            cv2.imwrite(f'/misc/data/private/autoPET/MM-WHS/mr/{mode}/label/slice_{n}.png', seg_slice)
            n += 1
    return n


def get_2d_images(image_path, label_path, name='ct'):
    n = 0
    for i in range(int(len(image_path) * 0.9)):

        nifti_img = nib.load(image_path[i])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[i])
        seg_3d = nifti_seg.get_fdata()
        if name == 'ct':
            n = ct_array2slices(img_3d, seg_3d, n, mode='train')
        elif name == 'mr':
            n = mr_array2slices(img_3d, seg_3d, n, mode='train')

    print("finished train data set")
    n = 0
    for j in range(int(len(image_path) * 0.9), int(len(image_path))):

        nifti_img = nib.load(image_path[j])
        img_3d = nifti_img.get_fdata()
        nifti_seg = nib.load(label_path[j])
        seg_3d = nifti_seg.get_fdata()
        if name == 'ct':
            n = ct_array2slices(img_3d, seg_3d, n, mode='val')
        elif name == 'mr':
            n = mr_array2slices(img_3d, seg_3d, n, mode='val')
    print("finished val data set")


def list_images(path):
    image_path = []
    label_path = []
    # read files names
    names = os.listdir(path)
    image_names = sorted(list(filter(lambda x: x.endswith('image.nii.gz'), names)))
    #label_names = list(filter(lambda x: x.endswith('label.nii.gz'), names))

    for i in range(len(image_names)):
        image_path.append(os.path.join(path, image_names[i]))
        label_path.append(os.path.join(path, image_names[i].replace('image.nii.gz', 'label.nii.gz')))

    return image_path, label_path


if __name__ == '__main__':
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/train/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/image', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/ct/val/label', exist_ok=True)
    os.makedirs('/misc/data/private/autoPET/ct_mr/mr/train/image', exist_ok=True)


    path_ct = "/misc/data/private/autoPET/MM-WHS-2017-Dataset/ct_train/"
    path_mr = "/misc/data/private/autoPET/MM-WHS-2017-Dataset/mr_train/"

    ct_image_paths, ct_label_paths = list_images(path_ct)
    mr_image_paths, mr_label_paths = list_images(path_mr)
    get_2d_images(ct_image_paths, ct_label_paths, name='ct')
    get_2d_images(mr_image_paths, mr_label_paths, name='mr')

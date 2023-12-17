import shutil
import sys
from pathlib import Path
import argparse
import subprocess
import os
import nibabel as nib
import numpy as np
from totalsegmentator.libs import combine_masks



masks = ["humerus"]


def move_images(path1, path2, num):
    for i in range(num):
        file_path = os.path.join(path1, f'slice_{i}.png')
        shutil.copyfile(file_path, os.path.join(path2, f'slice_{i}.png'))



def masks_combine(path):
    people_name = os.listdir(path)
    for item in people_name:
        print(item)
        input_dir = os.path.join(path, item)
        for mask in masks:
            print(mask)
            combined_img = combine_masks(input_dir, mask)
            output_dir = os.path.join(input_dir, f'{mask}.nii.gz')
            nib.save(combined_img, output_dir)


if __name__ == "__main__":
    root_path = '/data/private/autoPET/Task1/ct_label'
    path1 = '/misc/data/private/autoPET/CT_MR/mr/train'
    path2 = '/misc/data/private/autoPET/CT_MR/mr/val'
    move_images(path1, path2, 500)
    #masks_combine(root_path)

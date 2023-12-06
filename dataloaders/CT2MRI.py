import random
import torch
from torchvision import transforms as TR
from torchvision.transforms import functional
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.utils import data
import re


class CT2MRI(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics, for_supervision=False):

        opt.load_size = 128 if for_metrics else 128
        opt.crop_size = 128 if for_metrics else 128
        opt.label_nc = 8
        opt.contain_dontcare_label = True
        opt.semantic_nc = 9 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.opt = opt
        self.for_metrics = for_metrics
        self.for_supervision = False
        self.images, self.labels = self.list_images()

        if opt.mixed_images and not for_metrics:
            self.mixed_index = np.random.permutation(len(self))
        else:
            self.mixed_index = np.arange(len(self))

        if for_supervision:

            if opt.model_supervision == 0:
                return
            elif opt.model_supervision == 1:
                self.supervised_indecies = np.array(np.random.choice(len(self), opt.supervised_num), dtype=int)
            elif opt.model_supervision == 2:
                self.supervised_indecies = np.arange(len(self), dtype=int)
            images = []
            labels = []

            for index in self.supervised_indecies:
                images.append(self.images[index])
                labels.append(self.labels[index])

            self.images = images
            self.labels = labels

            self.mixed_index = np.arange(len(self))

            classes_counts = np.zeros((34), dtype=int)
            supervised_classes_in_images = []
            counts_in_images = []
            self.weights = []
            for i in tqdm(range(len(self))):
                label = self.__getitem__(i)['label']
                supervised_classes_in_image, counts_in_image = torch.unique(label, return_counts=True)
                supervised_classes_in_image = supervised_classes_in_image.int().numpy()
                counts_in_image = counts_in_image.int().numpy()
                supervised_classes_in_images.append(supervised_classes_in_image)
                counts_in_images.append(counts_in_image)
                for supervised_class_in_image, count_in_image in zip(supervised_classes_in_image, counts_in_image):
                    classes_counts[supervised_class_in_image] += count_in_image

            for i in range(len(self)):
                weight = 0
                for class_in_image, class_count_in_image in zip(supervised_classes_in_images[i], counts_in_images[0]):
                    if class_count_in_image != 0 and class_in_image != 0:
                        weight += class_in_image / classes_counts[class_in_image]

                self.weights.append(weight)

            min_weight = min(self.weights)
            self.weights = [weight / min_weight for weight in self.weights]
            self.for_supervision = for_supervision

    def __len__(self, ):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = Image.open(self.labels[idx])
        image, label = self.transforms(image, label)
        # label = np.asarray(label)
        if self.for_supervision:
            return {"image": image, "label": label, "name": self.images[self.mixed_index[idx]],
                    "weight": self.weights[self.mixed_index[idx]]}
        else:
            return {"image": image, "label": label, "name": self.images[self.mixed_index[idx]]}

    def list_images(self):
        mode = "val" if self.opt.phase == "test" or self.for_metrics else "train" #####val
        images = []
        labels = []
        path_img = os.path.join(self.opt.dataroot, 'mr', mode, "image")
        file_list_image = os.listdir(path_img)
        path_lab = os.path.join(self.opt.dataroot, 'ct', mode, "label")
        file_list_label = os.listdir(path_lab)
        sorted_file_list_image = sorted(file_list_image, key=lambda x: (int(x.split('_')[-1].split('.')[0])))
        sorted_file_list_label = sorted(file_list_label, key=lambda x: (int(x.split('_')[-1].split('.')[0])))
        for item in sorted_file_list_image:
            images.append(os.path.join(path_img, item))
        for item in sorted_file_list_label:
            labels.append(os.path.join(path_lab, item))
        #assert len(images) == len(labels), "different len of images and labels %s - %s" % (len(images), len(labels))

        return images, labels

    def transforms(self, image, label):
        # resize

        # flip
        if not (self.opt.phase == "test" or self.opt.phase != "train" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = TR.functional.hflip(image)
                label = TR.functional.hflip(label)
        # to tensor
        label = np.asarray(label).astype(np.uint8)
        pixel_mapping = {0: 0, 164: 1, 38: 2, 205: 3, 82: 4, 52: 5, 244: 6, 88: 7}
        # {0:0, 38:1, 52:2, 82:3, 88:4, 164:5, 205:6, 244:7}
        label = np.vectorize(pixel_mapping.get)(label)
        label = np.clip(label, 0, 255)
        ''''
        unique_values = set()
        pixels = label.flatten().tolist()
        for i in pixels:
            unique_values.add(i)
        print(unique_values)
        '''
        image = TR.functional.to_tensor(image)
        label = torch.from_numpy(label).to(torch.uint8)
        label = label.unsqueeze(0)
        # [3, 256, 256] [1, 256, 256]
        # normalize
        image = TR.functional.resize(image, [128, 128])
        label = TR.functional.resize(label, [128, 128])
        image = TR.functional.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image, label
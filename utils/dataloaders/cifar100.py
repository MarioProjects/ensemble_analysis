import torch
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations

import utils.dataloaders.utils as d


class CIFAR100Dataset(Dataset):
    """
    Dataset CIFAR100.
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(self, mode, transform, normalization="statistics", data_prefix=""):
        """
        :param mode: (string) Dataset mode in ["train", "validation"]
        :param transform: (list) List of albumentations applied to image and mask
        :param normalization: (str) Normalization mode. One of 'reescale', 'standardize', 'statistics'
        """

        if mode not in ["train", "validation", "test"]:
            assert False, "Unknown mode '{}'".format(mode)

        if normalization not in ['reescale', 'standardize', 'statistics']:
            assert False, "Unknown normalization '{}'".format(normalization)

        self.base_dir = os.path.join(data_prefix, "data", "CIFAR100")
        self.include_background = False
        self.img_channels = 3
        self.class_to_cat = {
            0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 6: 'bee', 7: 'beetle', 8: 'bicycle',
            9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can',
            17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud',
            24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 30: 'dolphin',
            31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house',
            38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard',
            45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom',
            52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck',
            59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit',
            66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark',
            74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 80: 'squirrel',
            81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone',
            87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle',
            94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'
        }
        self.num_classes = 100

        if mode == "train":
            data = np.load(os.path.join(self.base_dir, "x_train.npy"))
            labels = np.load(os.path.join(self.base_dir, "y_train.npy"))
            data = data[:int(len(data) * .90)]
            labels = labels[:int(len(labels) * .90)]
        elif mode == "validation":
            data = np.load(os.path.join(self.base_dir, "x_train.npy"))
            labels = np.load(os.path.join(self.base_dir, "y_train.npy"))
            data = data[int(len(data) * .90):]
            labels = labels[int(len(labels) * .90):]
        else:  # mode == test
            data = np.load(os.path.join(self.base_dir, "x_test.npy"))
            labels = np.load(os.path.join(self.base_dir, "y_test.npy"))

        self.labels = labels
        self.data = data
        self.mode = mode
        self.normalization = normalization

        self.transform = albumentations.Compose(transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image = self.data[idx]
        label = self.labels[idx]

        image, mask = d.apply_augmentations(image, self.transform, None, None)
        if self.normalization == "statistics":
            norm_transform = albumentations.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
            image = norm_transform(image=image)["image"]
        else:
            image = d.apply_normalization(image, self.normalization)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()

        return {"image": image, "label": label}

import torch
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations

import utils.dataloaders.utils as d


class CIFAR10Dataset(Dataset):
    """
    Dataset CIFAR10.
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

        self.base_dir = os.path.join(data_prefix, "data", "CIFAR10")
        self.include_background = False
        self.img_channels = 3
        self.class_to_cat = {
            0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
            5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck", 10: "Mean"
        }
        self.num_classes = 10

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

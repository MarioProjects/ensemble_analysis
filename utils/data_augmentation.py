""" --- DATA AUGMENTATION METHODS --- """

from RandAugment.augmentations import RandAugment, CutoutDefault
from torchvision.transforms import transforms


def data_augmentation_selector(da_policy, cutout_size=0, randaugment=False, n=0, m=0):
    """

    Args:
        da_policy:
        cutout_size: (int) If > 0 then apply cutout technique
        randaugment: (bool) Whether apply RandAugment or not [https://arxiv.org/pdf/1909.13719.pdf]
        n: (int) Number of augmentation transformations to apply sequentially.
        m: (int) Magnitude for all the transformations.

    Returns:

    """
    if da_policy == "cifar":
        return cifar_da(cutout_size, randaugment, n, m)

    assert False, "Unknown Data Augmentation Policy: {}".format(da_policy)


# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------- #


###########################################
#  --- DATA AUGMENTATION COMBINATIONS --- #
###########################################

def cifar_da(cutout_size=0, randaugment=False, n=0, m=0):
    """

    Args:
        cutout_size: (int) If > 0 then apply cutout technique
        randaugment: (bool) Whether apply RandAugment or not [https://arxiv.org/pdf/1909.13719.pdf]
        n: (int) Number of augmentation transformations to apply sequentially.
        m: (int) Magnitude for all the transformations.

    Returns:

    """
    print("Using CIFAR Data Augmentation Combinations")

    _CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    train_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])
    val_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])

    if randaugment:
        if n == 0 or m == 0:
            assert False, f"(RandAugment) Please, N and M should be greater than 0!"
        print("Applying RandAugment!")
        train_aug.transforms.insert(1, RandAugment(n, m))
    elif n != 0 or m != 0:
        assert False, f"You specified RandAugment arguments but do not use RandAugment flag!"

    if cutout_size > 0:
        train_aug.transforms.append(CutoutDefault(cutout_size))

    return train_aug, val_aug

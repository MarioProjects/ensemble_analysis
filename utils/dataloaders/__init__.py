from torch.utils.data import DataLoader

from utils.dataloaders.cifar10 import CIFAR10Dataset
from utils.dataloaders.cifar100 import CIFAR100Dataset


def dataset_selector(train_aug, val_aug, args, is_test=False, data_prefix=""):
    if args.dataset == "CIFAR10":
        if is_test:
            test_dataset = CIFAR10Dataset(
                mode="test", transform=val_aug, normalization=args.normalization, data_prefix=data_prefix
            )

            return DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False
            )

        train_dataset = CIFAR10Dataset(
            mode="train", transform=train_aug, normalization=args.normalization, data_prefix=data_prefix
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True
        )

        val_dataset = CIFAR10Dataset(
            mode="validation", transform=val_aug, normalization=args.normalization, data_prefix=data_prefix
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, drop_last=False
        )

    elif args.dataset == "CIFAR100":
        if is_test:
            test_dataset = CIFAR100Dataset(
                mode="test", transform=val_aug, normalization=args.normalization, data_prefix=data_prefix
            )

            return DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=False
            )

        train_dataset = CIFAR100Dataset(
            mode="train", transform=train_aug, normalization=args.normalization, data_prefix=data_prefix
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True
        )

        val_dataset = CIFAR100Dataset(
            mode="validation", transform=val_aug, normalization=args.normalization, data_prefix=data_prefix
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, drop_last=False
        )

    else:
        assert False, f"Unknown dataset '{args.dataset}'"

    print(f"Train dataset len:  {len(train_dataset)}")
    print(f"Validation dataset len:  {len(val_dataset)}")
    return train_loader, val_loader

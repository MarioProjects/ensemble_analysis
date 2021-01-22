import os
import torch

from utils.metrics import compute_accuracy


def get_logits_paths(logits_dir, prefix):
    paths = []
    for subdir, dirs, files in os.walk(logits_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if f"{prefix}" in file_path:
                paths.append(file_path)

    if not len(paths):
        assert False, f"Could not find any file at subdirectories of '{logits_dir}' with prefix '{prefix}'"

    return paths


def load_logits(logits_paths, get_accuracy=False):
    logits_list, labels_list, logits_names, logits_accuracy = [], [], [], []
    for lp in logits_paths:
        logits_name = "/".join(lp.split("/")[-2:])
        info = torch.load(lp, map_location=torch.device('cpu'))
        logits = info["logits"].cpu()
        labels = info["labels"].cpu()

        logits_list.append(logits)
        labels_list.append(labels)
        logits_names.append(logits_name)

        if get_accuracy:
            accuracy = compute_accuracy(labels, logits)
            logits_accuracy.append(accuracy)

    # logits_list shape: torch.Size([N, 10000, 10]) (CIFAR10 example)
    logits_list = torch.stack(logits_list)

    # -- Check if al labels has the same order for all logits --
    labels = labels_list[0]
    for indx, label_list in enumerate(labels_list[1:]):
        # Si alguno difiere del primero es que no es igual al resto tampoco
        if not torch.all(labels.eq(label_list)):
            assert False, f"Labels list does not match!"

    return logits_list, labels_list, logits_names, logits_accuracy


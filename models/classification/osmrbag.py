# https://github.com/osmr/imgclsmob/tree/master/pytorch
# https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/model_provider.py
# pip install pytorchcv
from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model


def osmr_selector(model_name, num_classes, in_channels):
    if "resnet18" in model_name:
        model = ptcv_get_model("resnet18", pretrained=True if "pretrained" in model_name else False)
        model.last_fc = nn.Linear(2048, num_classes)
    else:
        assert False, "Unknown model selected: {}".format(model_name)
    return model

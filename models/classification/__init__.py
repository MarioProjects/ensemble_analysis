from .osmrbag import *
from .cifar import *


def model_selector_classification(model_name, num_classes=2, in_channels=3):
    """

    :param model_name:
    :param num_classes:
    :param in_channels:
    :return:
    """

    if "osmr" in model_name:
        model = osmr_selector(model_name, num_classes, in_channels)

    elif "cifar" in model_name:
        model = cifar_selector(model_name, in_channels, num_classes)

    else:
        assert False, "Unknown model selected: {}".format(model_name)

    return model

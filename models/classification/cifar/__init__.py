from .shake_resnet import ShakeResNet
from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *
from .regnet import *
from .dla_simple import *
from .dla import *
from .wideresnet import WideResNet


def cifar_selector(model_name, in_channels, num_classes):
    
    if "resnet18" in model_name:
        model = ResNet18(in_channels, num_classes)
    elif "resnet34" in model_name:
        model = ResNet34(in_channels, num_classes)
    elif "resnet50" in model_name:
        model = ResNet50(in_channels, num_classes)
    elif "resnet101" in model_name:
        model = ResNet101(in_channels, num_classes)
    elif "resnet152" in model_name:
        model = ResNet152(in_channels, num_classes)

    elif "densenet121" in model_name:
        model = DenseNet121()
    elif "densenet161" in model_name:
        model = DenseNet161()
    elif "densenet" in model_name:
        model = densenet_cifar()

    elif 'wresnet28_10' in model_name:
        model = WideResNet(28, 10, dropout_rate=0.0, num_classes=num_classes)
    elif 'shakeshake26_2x96d' in model_name:
        model = ShakeResNet(26, 96, num_classes)
    
    else:
        assert False, "Unknown model selected: {}".format(model_name)
    return model

import torch
from torch.optim.swa_utils import AveragedModel

from models.classification import model_selector_classification


def model_selector(problem_type, model_name, num_classes, in_channels, checkpoint="", from_swa=False):
    """

    Args:
        problem_type:
        model_name:
        num_classes:
        in_channels:
        checkpoint:
        from_swa:

    Returns:

    """

    if problem_type == "classification":
        model = model_selector_classification(model_name, num_classes, in_channels)
    else:
        assert False, f"Unknown problem type '{problem_type}'"

    model_total_params = sum(p.numel() for p in model.parameters())
    print("Model total number of parameters: {}".format(model_total_params))
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if from_swa:
        model = AveragedModel(model)

    if checkpoint != "":
        print("Loaded model from checkpoint: {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint))

    return model

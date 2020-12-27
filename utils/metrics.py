import os
from utils.general import dict2df
import numpy as np
import torch

AVAILABLE_METRICS = ("accuracy")


class MetricsAccumulator:
    """
    Tendremos una lista de metricas que seran un dicccionario
    al hacer el print mostrara el promedio para cada metrica
    Ejemplo metrics 2 epochs con 2 clases:
       {"iou": [[0.8, 0.3], [0.9, 0.7]], "dice": [[0.76, 0.27], [0.88, 0.66]]}
    """

    def __init__(self, problem_type, metric_list):
        """

        Args:
            problem_type:
            metric_list:
        """
        if problem_type not in ["classification"]:
            assert False, f"Unknown problem type: '{problem_type}', please specify a valid one!"
        self.problem_type = problem_type

        if metric_list is None or not isinstance(metric_list, list):
            assert False, "Please, you need to specify a metric [list]"

        diffmetrics = np.setdiff1d(metric_list, AVAILABLE_METRICS)
        if len(diffmetrics):
            assert False, f"'{diffmetrics}' metric(s) not implemented."

        self.metric_list = metric_list
        self.metric_methods_args = {}
        self.metrics_helpers = {}
        self.metric_methods = self.__metrics_init__()
        self.metrics = {metric_name: [] for metric_name in metric_list}
        self.is_updated = True

    def __metrics_init__(self):
        metric_methods = []
        for metric_str in self.metric_list:
            if metric_str in ["accuracy"]:
                self.metric_methods_args[metric_str] = {}
                metric_methods.append(compute_accuracy)
                self.metrics_helpers["accuracy_best_method"] = "max"
                self.metrics_helpers["accuracy_best_value"] = -1

        return metric_methods

    def record(self, prediction, target):

        if self.is_updated:
            for key in self.metrics:
                self.metrics[key].append([])
            self.is_updated = False

        for indx, metric in enumerate(self.metric_methods):
            self.metrics[self.metric_list[indx]][-1] += [
                metric(target, prediction, **self.metric_methods_args[self.metric_list[indx]])]

    def update(self):
        """
        CALL THIS METHOD AFTER RECORD ALL SAMPLES / AFTER EACH EPOCH
        We have accumulated metrics along different samples/batches and want to average accross that same epoch samples:
        {'accuracy': [[[0.8, 0.6, 0.3, 0.5]]]} -> {'accuracy': [[0.55]]}
        """
        for key in self.metrics:

            self.metrics[key][-1] = np.mean(self.metrics[key][-1])
            mean_metric_value = self.metrics[key][-1]

            if self.metrics_helpers[f"{key}_best_value"] < mean_metric_value:
                self.metrics_helpers[f"{key}_best_value"] = mean_metric_value
                self.metrics_helpers[f"{key}_is_best"] = True
            else:
                self.metrics_helpers[f"{key}_is_best"] = False

        self.is_updated = True

    def report_best(self):
        for key in self.metrics:
            print("\t- {}: {}".format(key, self.metrics_helpers[f"{key}_best_value"]))

    def mean_value(self, metric_name):
        return self.metrics[metric_name][-1][-1]

    def __str__(self, precision=3):
        output_str = ""
        for metric_key in self.metric_list:
            output_str += '{:{align}{width}.{prec}f} | '.format(
                self.metrics[metric_key][-1], align='^', width=len(metric_key), prec=3
            )

        return output_str


def compute_accuracy(y_true, y_pred):
    # Standard accuracy metric computed over all classes
    _, predicted = torch.max(y_pred.data, 1)
    total = y_true.size(0)
    correct = (predicted == y_true).sum().item()
    return correct / total

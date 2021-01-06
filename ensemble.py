#!/usr/bin/env python
# coding: utf-8

import pretty_errors

# ---- My utils ----
from utils.logits import *
from utils.neural import *
from utils.metrics import compute_accuracy
from utils.calibration import compute_calibration_metrics

pretty_errors.mono()


def ensemble_evaluation(logits_dir="logits", prefix="test", ensemble_strategy=None):
    # Check ensemble strategies are fine
    available_strategies = ["avg", "vote"]
    if ensemble_strategy is None or len(ensemble_strategy) == 0:
        assert False, "Please specify a ensemble strategy"
    ensemble_strategy = [ensemble_strategy] if isinstance(ensemble_strategy, str) else ensemble_strategy
    for strategy in ensemble_strategy:
        if strategy not in available_strategies:
            assert False, f"Unknown strategy {strategy}"

    # Get logits paths
    logits_paths = get_logits_paths(logits_dir, prefix)

    # Get logits and labels
    logits_list, labels_list, logits_names, logits_accuracy = load_logits(logits_paths, get_accuracy=True)
    labels = labels_list[0]

    # Compute calibration metrics
    softmax = nn.Softmax(dim=2)
    probs_list = softmax(logits_list)

    logits_calibration_metrics = []
    for prob_list in probs_list:
        ece, mce, brier, nnl = compute_calibration_metrics(prob_list, labels, apply_softmax=False, bins=15)
        logits_calibration_metrics.append({"ece": ece, "mce": mce, "brier": brier, "nnl": nnl})

    # -- Display Results --
    width_methods = max(len("Avg probs ensemble"), max([len(x) for x in logits_names]))

    header = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
        "Method", "Accuracy", "ECE", "MCE", "BRIER", "NNL", align='^', widthL=width_methods, widthA=8, widthM=6,
    )
    print("".join(["_"] * len(header)))
    print(header)
    print("".join(["_"] * len(header)))
    for indx, logit_name in enumerate(logits_names):
        line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
            logit_name, logits_accuracy[indx],
            logits_calibration_metrics[indx]["ece"], logits_calibration_metrics[indx]["mce"],
            logits_calibration_metrics[indx]["brier"], logits_calibration_metrics[indx]["nnl"],
            align='^', widthL=width_methods, widthA=8, widthM=6,
        )
        print(line)

    # -- Ensemble Strategies --
    if "avg" in ensemble_strategy:
        probs_avg = probs_list.sum(dim=0) / len(probs_list)
        probs_avg_accuracy = compute_accuracy(labels, probs_avg)
        ece, mce, brier, nnl = compute_calibration_metrics(probs_avg, labels, apply_softmax=False, bins=15)
        line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
            "Avg probs ensemble", probs_avg_accuracy, ece, mce, brier, nnl,
            align='^', widthL=width_methods, widthA=8, widthM=6,
        )
        print(line)

    if "vote" in ensemble_strategy:
        _, vote_list = torch.max(logits_list.data, dim=2)
        vote_list = torch.nn.functional.one_hot(vote_list)
        vote_list = vote_list.sum(dim=0)
        _, vote_list = torch.max(vote_list.data, dim=1)
        vote_accuracy = (vote_list == labels).sum().item() / len(labels)
        line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
            "Vote ensemble", vote_accuracy, "----", "----", "----", "----",
            align='', widthL=width_methods, widthA=8, widthM=6,
        )
        print(line)

    print("".join(["_"] * len(header)))


print("\n---- Validation evaluation ----\n")
ensemble_evaluation("logits", prefix="val", ensemble_strategy=["avg"])

print("\n---- Test evaluation ----\n")
ensemble_evaluation("logits", prefix="test", ensemble_strategy=["avg"])

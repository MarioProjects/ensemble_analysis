#!/usr/bin/env python
# coding: utf-8

# Usage: python matrix_scaling.py --epochs 250 --scheduler_steps 70 125 180 220 --logits_dir logits_res18_cifar10

import pretty_errors

# ---- My utils ----
from utils.logits import *
from utils.neural import *
from utils.metrics import compute_accuracy
from utils.calibration import compute_calibration_metrics
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

import argparse


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(description='Matrix Scaling Analysis', formatter_class=SmartFormatter)

parser.add_argument('--verbose', action='store_true', help='Display or not matrix learning process')
parser.add_argument('--epochs', type=int, default=100, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for training')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
parser.add_argument('--logits_dir', type=str, default="logits", help='Logits directory')
parser.add_argument('--scheduler_steps', '--arg', nargs='+', type=int, help='Steps for learning rate decay')
args = parser.parse_args()

pretty_errors.mono()
verbose = args.verbose  # Display or not matrix learning process

# ---- Load logits ----
logits_dir = args.logits_dir

# -> Validation Logits
prefix = "val_logits"
val_logits_paths = get_logits_paths(logits_dir, prefix)
val_logits_list, val_labels_list, val_logits_names, _ = load_logits(val_logits_paths, get_accuracy=False)
val_labels = val_labels_list[0]

# -> Validation Avg Ensemble
prefix = "val_avg_ensemble"
val_avg_ensemble_logits_paths = get_logits_paths(logits_dir, prefix)
val_avg_ensemble_logits_list, val_avg_ensemble_labels_list, val_avg_ensemble_logits_names, _ = load_logits(
    val_avg_ensemble_logits_paths, get_accuracy=False
)
val_avg_ensemble_labels = val_avg_ensemble_labels_list[0]

if not (val_avg_ensemble_labels == val_labels).all():
    assert False, "Validation logits and Validation ensemble logits should be equal!"

# -> Test Logits
prefix = "test_logits"
test_logits_paths = get_logits_paths(logits_dir, prefix)
test_logits_list, test_labels_list, test_logits_names, _ = load_logits(test_logits_paths, get_accuracy=False)
test_labels = test_labels_list[0]

# -> Test Avg Ensemble
prefix = "test_avg_ensemble"
test_avg_ensemble_logits_paths = get_logits_paths(logits_dir, prefix)
test_avg_ensemble_logits_list, test_avg_ensemble_labels_list, test_avg_ensemble_logits_names, _ = load_logits(
    test_avg_ensemble_logits_paths, get_accuracy=False
)
test_avg_ensemble_labels = test_avg_ensemble_labels_list[0]


# ---- matrix SCALING ----
# https://github.com/gpleiss/matrix_scaling/blob/master/matrix_scaling.py
class MatrixScaling(nn.Module):
    """
    A thin decorator, which wraps a model with matrix scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits, NOT the softmax (or log softmax)!
    """

    def __init__(self, logits_size):
        super(MatrixScaling, self).__init__()
        self.matrix = nn.Parameter(torch.ones(logits_size))

    def forward(self, model_logits):
        return self.matrix * model_logits


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def learn_matrix(model_logits, model_labels):
    # Training parameters
    criterion = nn.CrossEntropyLoss().cuda()
    if args.scheduler_steps is None:
        scheduler_steps = np.arange(0, args.epochs, args.epochs // 5)

    # Create 1 matrix parameter per model / val logits
    matrix = MatrixScaling(model_logits.shape[1])
    optimizer = SGD(matrix.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler_steps, gamma=0.1)

    if verbose:
        header = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthA}} | {:{align}{widthLL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
            "Epoch", "LR", "Loss", "Temp Param", "Accuracy", "ECE", "MCE", "BRIER", "NNL",
            align='^', widthL=8, widthLL=10, widthA=8, widthM=6,
        )

        print("".join(["_"] * len(header)))
        print(header)
        print("".join(["_"] * len(header)))

    for epoch in range(args.epochs):

        matrix.train()
        train_loss, correct, total = [], 0, 0
        c_ece, c_mce, c_brier, c_nnl = [], [], [], []

        for c_logits, c_labels in zip(chunks(model_logits, args.batch_size), chunks(model_labels, args.batch_size)):
            # Train
            optimizer.zero_grad()
            new_logits = matrix(c_logits)
            loss = criterion(new_logits, c_labels)
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss.append(loss.item())
            _, predicted = new_logits.max(1)
            total += len(c_labels)
            correct += predicted.eq(c_labels).sum().item()

            softmax = nn.Softmax(dim=1)
            new_probs_list = softmax(new_logits)
            ece, mce, brier, nnl = compute_calibration_metrics(new_probs_list, c_labels, apply_softmax=False, bins=15)
            c_ece.append(ece)
            c_mce.append(mce)
            c_brier.append(brier.item())
            c_nnl.append(nnl.item())

        c_train_loss = np.array(train_loss).mean()
        c_accuracy = correct / total
        c_ece = np.array(c_ece).mean()
        c_mce = np.array(c_mce).mean()
        c_brier = np.array(c_brier).mean()
        c_nnl = np.array(c_nnl).mean()
        current_lr = get_current_lr(optimizer)

        if verbose:
            line = "| {:{align}{widthL}} | {:{align}{widthA}.6f} | {:{align}{widthA}.4f} | {:{align}{widthLL}.4f} | {:{align}{widthA}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
                epoch + 1, current_lr, c_train_loss, matrix.matrix.item(), c_accuracy, c_ece, c_mce,
                c_brier,
                c_nnl,
                align='^', widthL=8, widthA=8, widthM=6, widthLL=10
            )
            print(line)

        scheduler.step()

    return matrix


matrices_val = []
print(f"Validation Logits -> Calculating matrix for {len(val_logits_list)} models...")
for indx, val_model_logits in enumerate(val_logits_list):
    matrices_val.append(learn_matrix(val_model_logits, val_labels))
    print(f"Model {indx} done!")
print("-- Finished --\n")

matrices_avg_ensemble = []
print(f"Validation Logits Ensemble Avg -> Calculating matrix for {len(val_avg_ensemble_logits_list)} models...")
for indx, val_model_logits in enumerate(val_avg_ensemble_logits_list):
    matrices_avg_ensemble.append(learn_matrix(val_model_logits, val_labels))
    print(f"Ensemble Avg {indx} done!")
print("-- Finished --\n")


# ---- Display Results ----
def display_results(logits_names, logits_list, labels, matrices, avg=True, get_logits=False):
    softmax = nn.Softmax(dim=1)
    width_methods = max(len("Avg probs ensemble"), max([len(x) for x in logits_names]))

    header = "\n| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
        "Method", "Accuracy", "ECE", "MCE", "BRIER", "NNL", align='^', widthL=width_methods, widthA=8, widthM=6,
    )
    print("".join(["_"] * len(header)))
    print(header)
    print("".join(["_"] * len(header)))
    probs_list, t_logits = [], []
    for indx, logit_name in enumerate(logits_names):
        # Scale with learned matrix parameter the logits
        matrices[indx].eval()
        logits = matrices[indx](logits_list[indx])
        t_logits.append(logits)
        # Compute metrics
        accuracy = compute_accuracy(labels, logits)
        probs = softmax(logits)
        probs_list.append(probs)
        ece, mce, brier, nnl = compute_calibration_metrics(probs, labels, apply_softmax=False, bins=15)
        # Display
        line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
            logit_name, accuracy, ece, mce, brier, nnl, align='^', widthL=width_methods, widthA=8, widthM=6,
        )
        print(line)

    # ---- Ensemble Strategies: Average ----
    if avg:
        probs_list = torch.stack(probs_list)
        probs_avg = probs_list.sum(dim=0) / len(probs_list)
        probs_avg_accuracy = compute_accuracy(labels, probs_avg)
        ece, mce, brier, nnl = compute_calibration_metrics(probs_avg, labels, apply_softmax=False, bins=15)
        line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
            "Avg probs ensemble", probs_avg_accuracy, ece, mce, brier, nnl,
            align='^', widthL=width_methods, widthA=8, widthM=6,
        )
        print(line)
    if get_logits:
        return torch.stack(t_logits)


# --- Avg ensemble
val_cal_logits = display_results(val_logits_names, val_logits_list, val_labels, matrices_val, get_logits=True)
val_cal_logits_ensemble = val_cal_logits.detach().sum(dim=0)
test_cal_logits = display_results(test_logits_names, test_logits_list, test_labels, matrices_val, get_logits=True)
test_cal_logits_ensemble = test_cal_logits.detach().sum(dim=0)

# --- Avg ensemble T
print("\n\n--- Avg ensemble T ---")
display_results(
    val_avg_ensemble_logits_names, val_avg_ensemble_logits_list, val_labels, matrices_avg_ensemble, avg=False
)
display_results(
    test_avg_ensemble_logits_names, test_avg_ensemble_logits_list, test_labels, matrices_avg_ensemble, avg=False
)

# --- Avg ensemble CT
print("\n\n--- Avg ensemble CT ---")
matrices_avg_ensemble = []
val_ct_temp = [learn_matrix(val_cal_logits_ensemble, val_labels)]

display_results(
    ["val_ct_avg_ensemble_logits"], val_cal_logits_ensemble.unsqueeze(0), val_labels, val_ct_temp, avg=False
)
display_results(
    ["test_ct_avg_ensemble_logits"], test_cal_logits_ensemble.unsqueeze(0), test_labels, val_ct_temp, avg=False
)

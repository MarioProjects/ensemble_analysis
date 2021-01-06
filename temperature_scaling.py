#!/usr/bin/env python
# coding: utf-8

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


parser = argparse.ArgumentParser(description='Temperature Scaling Analysis', formatter_class=SmartFormatter)

parser.add_argument('--verbose', action='store_true', help='Display or not temperature learning process')
parser.add_argument('--epochs', type=int, default=100, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for training')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
parser.add_argument('--scheduler_steps', '--arg', nargs='+', type=int, help='Steps for learning rate decay')
args = parser.parse_args()

pretty_errors.mono()
verbose = args.verbose  # Display or not temperature learning process

# ---- Load logits ----
logits_dir = "logits"

# -> Validation Logits
prefix = "val"
val_logits_paths = get_logits_paths(logits_dir, prefix)

# -> Test Logits
prefix = "test"
test_logits_paths = get_logits_paths(logits_dir, prefix)

# ---- Load logits and labels ----

# -> Validation

val_logits_list, val_labels_list, val_logits_names, _ = load_logits(val_logits_paths, get_accuracy=False)
val_labels = val_labels_list[0]

# -> Test
test_logits_list, test_labels_list, test_logits_names, _ = load_logits(test_logits_paths, get_accuracy=False)
test_labels = test_labels_list[0]


# ---- TEMPERATURE SCALING ----
# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
class TempScaling(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits, NOT the softmax (or log softmax)!
    """

    def __init__(self):
        super(TempScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, model_logits):
        return self.temperature_scale(model_logits)

    def temperature_scale(self, model_logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(model_logits.size(0), model_logits.size(1))
        return model_logits * temperature


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Training parameters
lr = args.learning_rate
batch_size = args.batch_size
criterion = nn.CrossEntropyLoss().cuda()

if args.scheduler_steps is None:
    args.scheduler_steps = np.arange(0, args.epochs, args.epochs//5)

# Create 1 temperature parameter per model / val logits
temperatures = [TempScaling() for i in range(len(val_logits_list))]
optimizers = [SGD(temperature.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for temperature in temperatures]
schedulers = [MultiStepLR(optimizer, milestones=args.scheduler_steps, gamma=0.1) for optimizer in optimizers]

print(f"\n{len(temperatures)} Models. Searching Temperatures...")
for indx, (temperature, optimizer, scheduler) in enumerate(zip(temperatures, optimizers, schedulers)):
    val_model_logits = val_logits_list[indx]

    if verbose:
        header = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthA}} | {:{align}{widthLL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
            "Epoch", "LR", "Loss", "Temp Param", "Accuracy", "ECE", "MCE", "BRIER", "NNL",
            align='^', widthL=8, widthLL=10, widthA=8, widthM=6,
        )

        print("".join(["_"] * len(header)))
        print(header)
        print("".join(["_"] * len(header)))

    for epoch in range(args.epochs):

        temperature.train()
        train_loss, correct, total = [], 0, 0
        c_ece, c_mce, c_brier, c_nnl = [], [], [], []

        for c_logits, c_labels in zip(chunks(val_model_logits, batch_size), chunks(val_labels, batch_size)):
            # Train
            optimizer.zero_grad()
            new_logits = temperature(c_logits)
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
                epoch + 1, current_lr, c_train_loss, temperature.temperature.item(), c_accuracy, c_ece, c_mce, c_brier,
                c_nnl,
                align='^', widthL=8, widthA=8, widthM=6, widthLL=10
            )
            print(line)

        scheduler.step()

    print(f"Model {indx + 1} done!")


# ---- Display Results ----
def display_results(logits_names, logits_list, labels):
    softmax = nn.Softmax(dim=1)
    width_methods = max(len("Avg probs ensemble"), max([len(x) for x in logits_names]))

    header = "\n| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
        "Method", "Accuracy", "ECE", "MCE", "BRIER", "NNL", align='^', widthL=width_methods, widthA=8, widthM=6,
    )
    print("".join(["_"] * len(header)))
    print(header)
    print("".join(["_"] * len(header)))
    probs_list = []
    for indx, logit_name in enumerate(logits_names):
        # Scale with learned temperature parameter the logits
        temperatures[indx].eval()
        logits = temperatures[indx](logits_list[indx])
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
    probs_list = torch.stack(probs_list)
    probs_avg = probs_list.sum(dim=0) / len(probs_list)
    probs_avg_accuracy = compute_accuracy(labels, probs_avg)
    ece, mce, brier, nnl = compute_calibration_metrics(probs_avg, labels, apply_softmax=False, bins=15)
    line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
        "Avg probs ensemble", probs_avg_accuracy, ece, mce, brier, nnl,
        align='^', widthL=width_methods, widthA=8, widthM=6,
    )
    print(line)


display_results(val_logits_names, val_logits_list, val_labels)
display_results(test_logits_names, test_logits_list, test_labels)

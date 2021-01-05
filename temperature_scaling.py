#!/usr/bin/env python
# coding: utf-8

import pretty_errors

# ---- My utils ----
from utils.neural import *
from utils.metrics import compute_accuracy
from utils.calibration import compute_calibration_metrics
from torch.optim import SGD
from torch.optim.lr_scheduler import  MultiStepLR

pretty_errors.mono()
verbose = False

# ---- Load logits ----
logits_dir = "logits"

# -> Validation Logits

prefix = "val"

val_logits_paths = []
for subdir, dirs, files in os.walk(logits_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if f"{prefix}_logits" in file_path:
            val_logits_paths.append(file_path)

if not len(val_logits_paths):
    assert False, f"Could not find any file at subdirectoreis of '{logits_dir}' with prefix '{prefix}'"

# -> Test Logits

prefix = "test"

test_logits_paths = []
for subdir, dirs, files in os.walk(logits_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if f"{prefix}_logits" in file_path:
            test_logits_paths.append(file_path)

if not len(test_logits_paths):
    assert False, f"Could not find any file at subdirectories of '{logits_dir}' with prefix '{prefix}'"

# ---- Get logits and labels ----

# -> Validation

val_logits_list, val_logits_names, val_labels_list = [], [], []
for lp in val_logits_paths:
    logits_name = "/".join(lp.split("/")[-2:])
    info = torch.load(lp)
    logits = info["logits"].cpu()
    labels = info["labels"].cpu()

    val_logits_list.append(logits)
    val_labels_list.append(labels)
    val_logits_names.append(logits_name)

# logits_list shape: torch.Size([N, 10000, 10]) (CIFAR10 example)
val_logits_list = torch.stack(val_logits_list)

# Check if al labels has the same order for all logits
val_labels = val_labels_list[0]
for indx, label_list in enumerate(val_labels_list[1:]):
    # Si alguno difiere del primero es que no es igual al resto tampoco
    if not torch.all(val_labels.eq(label_list)):
        assert False, f"Labels list does not match!"

# -> Test

test_logits_list, test_logits_names, test_labels_list = [], [], []
for lp in test_logits_paths:
    logits_name = "/".join(lp.split("/")[-2:])
    info = torch.load(lp)
    logits = info["logits"].cpu()
    labels = info["labels"].cpu()

    test_logits_list.append(logits)
    test_labels_list.append(labels)
    test_logits_names.append(logits_name)

# logits_list shape: torch.Size([N, 10000, 10]) (CIFAR10 example)
test_logits_list = torch.stack(test_logits_list)

# Check if al labels has the same order for all logits
test_labels = test_labels_list[0]
for indx, label_list in enumerate(test_labels_list[1:]):
    # Si alguno difiere del primero es que no es igual al resto tampoco
    if not torch.all(labels.eq(label_list)):
        assert False, f"Labels list does not match!"


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

    def forward(self, logits):
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits * temperature


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


lr = 0.1
n_epochs = 100
batch_size = 128
criterion = nn.CrossEntropyLoss().cuda()

temperatures = [TempScaling() for i in range(len(val_logits_list))]  # Create 1 parameter per model / val logits
optimizers = [SGD(temperature.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4) for temperature in temperatures]
schedulers = [MultiStepLR(optimizer, milestones=np.arange(0, 100, 20), gamma=0.1) for optimizer in optimizers]

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

    for epoch in range(n_epochs):

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
                epoch + 1, current_lr, c_train_loss, temperature.temperature.item(), c_accuracy, c_ece, c_mce, c_brier, c_nnl,
                align='^', widthL=8, widthA=8, widthM=6, widthLL=10
            )
            print(line)

        scheduler.step()

    print(f"Model {indx+1} done!")

# ---- Display Results: Validation ----
print("\n---- Validation evaluation ----\n")
softmax = nn.Softmax(dim=1)
width_methods = max(len("Avg probs ensemble"), max([len(x) for x in val_logits_names]))

header = "\n| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
    "Method", "Accuracy", "ECE", "MCE", "BRIER", "NNL", align='^', widthL=width_methods, widthA=8, widthM=6,
)
print("".join(["_"] * len(header)))
print(header)
print("".join(["_"]*len(header)))
val_probs_list = []
for indx, logit_name in enumerate(val_logits_names):
    # Scale with learned temperature parameter the logits
    temperatures[indx].eval()
    logits = temperatures[indx](val_logits_list[indx])
    # Compute metrics
    accuracy = compute_accuracy(val_labels, logits)
    probs = softmax(logits)
    val_probs_list.append(probs)
    ece, mce, brier, nnl = compute_calibration_metrics(probs, val_labels, apply_softmax=False, bins=15)
    # Display
    line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
            logit_name, accuracy, ece, mce, brier, nnl, align='^', widthL=width_methods, widthA=8, widthM=6,
    )
    print(line)

# ---- Ensemble Strategies: Average ----
val_probs_list = torch.stack(val_probs_list)
val_probs_avg = val_probs_list.sum(dim=0) / len(val_probs_list)
probs_avg_accuracy = compute_accuracy(val_labels, val_probs_avg)
ece, mce, brier, nnl = compute_calibration_metrics(val_probs_avg, val_labels, apply_softmax=False, bins=15)
line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
       "Avg probs ensemble", probs_avg_accuracy, ece, mce, brier, nnl,
       align='^', widthL=width_methods, widthA=8, widthM=6,
)
print(line)


# ---- Display Results: Test ----
print("\n---- Test evaluation ----\n")
softmax = nn.Softmax(dim=1)
width_methods = max(len("Avg probs ensemble"), max([len(x) for x in test_logits_names]))

header = "\n| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} | {:{align}{widthM}} |".format(
    "Method", "Accuracy", "ECE", "MCE", "BRIER", "NNL", align='^', widthL=width_methods, widthA=8, widthM=6,
)
print("".join(["_"] * len(header)))
print(header)
print("".join(["_"]*len(header)))
test_probs_list = []
for indx, logit_name in enumerate(test_logits_names):
    # Scale with learned temperature parameter the logits
    temperatures[indx].eval()
    logits = temperatures[indx](test_logits_list[indx])
    # Compute metrics
    accuracy = compute_accuracy(test_labels, logits)
    probs = softmax(logits)
    test_probs_list.append(probs)
    ece, mce, brier, nnl = compute_calibration_metrics(probs, test_labels, apply_softmax=False, bins=15)
    # Display
    line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
            logit_name, accuracy, ece, mce, brier, nnl, align='^', widthL=width_methods, widthA=8, widthM=6,
    )
    print(line)

# ---- Ensemble Strategies: Average ----
test_probs_list = torch.stack(test_probs_list)
test_probs_avg = test_probs_list.sum(dim=0) / len(test_probs_list)
probs_avg_accuracy = compute_accuracy(test_labels, test_probs_avg)
ece, mce, brier, nnl = compute_calibration_metrics(test_probs_avg, test_labels, apply_softmax=False, bins=15)
line = "| {:{align}{widthL}} | {:{align}{widthA}} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} | {:{align}{widthM}.4f} |".format(
       "Avg probs ensemble", probs_avg_accuracy, ece, mce, brier, nnl,
       align='^', widthL=width_methods, widthA=8, widthM=6,
)
print(line)
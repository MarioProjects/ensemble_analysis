#!/usr/bin/env python
# coding: utf-8

# ---- My utils ----
from models import model_selector
from utils.arguments import *
from utils.data_augmentation import data_augmentation_selector
from utils.datasets import dataset_selector
from utils.neural import *

_, val_aug = data_augmentation_selector(
    args.data_augmentation, args.img_size, args.crop_size
)
test_loader = dataset_selector(None, val_aug, args, is_test=True)

model = model_selector(
    args.problem_type, args.model_name, test_loader.dataset.num_classes, from_swa=args.swa_checkpoint,
    in_channels=test_loader.dataset.img_channels, devices=args.gpu, checkpoint=args.model_checkpoint
)

test_metrics = MetricsAccumulator(
    args.problem_type, args.metrics
)

test_metrics, test_logits, test_labels = val_step(
    test_loader, model, test_metrics,
)

torch.save(
    {"logits": test_logits, "labels": test_labels, "config": args},
    args.output_dir + f"/test_logits_{args.model_checkpoint.split('/')[-1]}"
)

print("\nResults:")
test_metrics.report_best()

if args.notify:
    slack_message(
        message=f"(Seed {args.seed}) {args.dataset.upper()} evaluation experiments finished!",
        channel="ensembles_analysis"
    )
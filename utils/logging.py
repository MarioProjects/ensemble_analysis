# https://pyformat.info/

from utils.general import current_time


def build_header(metric_list, display=True):
    """

    Args:
        metric_list: Example ["iou", "dice", "assd", "hausdorff"]
        display: Whether print header or not

    Returns:

    """

    header = "| {:{align}{widthL}} | {:{align}{width}} | {:{align}{widthI}} | {:{align}{widthLr}} | ".format(
        "Time", "epoch", "Info", "LRate", align='^', widthL=19, width=7, widthI=6, widthLr=10, prec1=5
    )

    for metric_str in metric_list:
        len_metric = len(metric_str)
        header += "{:{align}{width}.{prec1}} | ".format(
            metric_str, align='^', width=len_metric, prec1=len_metric,
        )

    if display:
        print("\n------------------------------- START TRAINING -------------------------------\n")
        print(header)
        print(*["="] * (len(header) // 2))

    return header


def log_epoch(current_epoch, current_lr, train_metrics, val_metrics, header):
    print("| {:{align}{widthL}} | {:{align}{width}} | {:{align}{widthI}} | {:{align}{widthLr}.{prec6}f} | {}".format(
        current_time(), current_epoch, "Train", current_lr, train_metrics,
        align='^', widthL=19, width=7, widthI=6, widthLr=10, prec6=6,
    ))

    print("| {:{align}{widthL}} | {:{align}{width}} | {:{align}{widthI}} | {:{align}{widthLr}} | {}".format(
        "", "", "Valid", "", val_metrics, align='^', widthL=19, width=7, widthI=6, widthLr=10,
    ))

    print(*["-"] * (len(header) // 2))

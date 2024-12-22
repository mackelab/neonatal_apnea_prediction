import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from sklearn.metrics import roc_auc_score, roc_curve


def plot_auroc_with_uncertainty(
    score_ls,
    label_ls,
    color="green",
    label="",
    linestyle="-",
    alpha_traces=0.0,
    alpha_uncertainty=0.1,
    auroc_decimals=2,
):
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    auc_ls = []
    for scores, labels in zip(score_ls, label_ls):
        fpr, tpr, _ = roc_curve(labels, scores)
        auc_ls.append(roc_auc_score(labels, scores))
        plt.plot(fpr, tpr, color=color, alpha=alpha_traces)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(
        base_fpr,
        mean_tprs,
        color,
        alpha=1.0,
        linestyle=linestyle,
        label=f"{label}{np.round(np.mean(auc_ls), decimals=auroc_decimals)} / {np.round(np.std(auc_ls), decimals=auroc_decimals)}",
    )
    plt.fill_between(
        base_fpr, tprs_lower, tprs_upper, color=color, alpha=alpha_uncertainty
    )

    return np.round(np.mean(auc_ls), decimals=auroc_decimals), np.round(
        np.std(auc_ls), decimals=auroc_decimals
    )


def plot_class_activation(
    input,
    class_activation,
    ax=None,
    fig=None,
    lower_y=-5.0,
    upper_y=5.0,
    supress_colorbar=False,
    x_ticks=None,
    y_ticks=None,
):
    assert len(input) == len(class_activation)
    steps = np.arange(len(input))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    points = np.array([steps, input]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(-3.0, 3.0)
    lc = LineCollection(segments, cmap="cool", norm=norm)
    lc.set_array(class_activation)
    lc.set_linewidth(1)
    line = ax.add_collection(lc)
    if not supress_colorbar:
        fig.colorbar(line, ax=ax)

    ax.set_xlim(steps.min(), steps.max())
    ax.set_ylim(lower_y, upper_y)

    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)


def get_gam_cams(time_window_idx, single_id_dataset, classifier):
    modalities = single_id_dataset.signal_types

    cams = {}
    signals = {}
    x_s, y_s = single_id_dataset._get_list(time_window_idx)

    logits, bias = classifier.compute_logits([val.unsqueeze(0) for val in x_s])
    score = classifier.forward([val.unsqueeze(0) for val in x_s])

    for typ, mod, mult, val in zip(
        modalities, classifier.module_list, classifier.multiplier_list, x_s
    ):
        with torch.no_grad():
            cam = mult * mod.class_act(val.unsqueeze(0))
            cam = cam.numpy().flatten()
            cams[typ] = cam
            signals[typ] = val.numpy().flatten()
    return (
        cams,
        signals,
        y_s.item(),
        [logit.item() for logit in logits],
        bias.item(),
        score.item(),
    )

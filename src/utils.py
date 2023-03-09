import os
import pickle
from copy import deepcopy
from datetime import datetime
from itertools import product
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Sampler, WeightedRandomSampler


def get_current_time_stamp_minutes():
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M")
    return current_time


def save_pickle(obj, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        raise FileExistsError(f"{file_path} already exists!")

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle_file(file_name, folder_path):
    file_path = os.path.join(folder_path, file_name)
    assert file_path.endswith(".pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def tensor_to_array(tensor):
    return tensor.cpu().detach().numpy()


# C1==C2 is not garantueed. Only in expectation.
class BalancedSampler(WeightedRandomSampler):
    def __init__(self, data_source, replacement=True):
        labels = data_source._get_labels()

        classes, counts = np.unique(labels, return_counts=True)
        weights = np.zeros_like(labels, dtype=float)
        for val, nums in zip(classes, counts):
            weights[labels == val] = 1.0 / nums
        num_samples = int(min(counts) * len(counts))

        super().__init__(
            weights=weights, num_samples=num_samples, replacement=replacement
        )

    def __iter__(self):
        return super().__iter__()

    def __len__(self):
        return super().__len__()


class MultiDatasetBalancedSampler(Sampler):
    def __init__(self, concat_data_source, replacement=True):
        offset = 0
        self.all_indices = []
        for dat in concat_data_source.datasets:
            indices = list(BalancedSampler(dat, replacement=replacement).__iter__())
            offset_indicies = [idx + offset for idx in indices]
            self.all_indices.extend(offset_indicies)
            offset += len(dat)

        shuffle(self.all_indices)

    def __iter__(self):
        yield from iter(self.all_indices)

    def __len__(self):
        return len(self.all_indices)


### Nested CV utils
def create_sweep_cfgs(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    sg_keys = find_sweep_grids(cfg_dict)
    sg_values = find_sg_values(cfg_dict, sg_key_tups=sg_keys)

    sweep_dicts = []
    for comb in product(*sg_values):
        new_dic = deepcopy(cfg_dict)
        for sg_key, sg_val in zip(sg_keys, comb):
            change_value(new_dic, sg_key, sg_val)
        sweep_dicts.append(new_dic)

    return [OmegaConf.create(dic) for dic in sweep_dicts]


def change_value(d, keys, new_value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = new_value


def find_sweep_grids(cfg_dict, carry_tup=()):
    final_list = []
    for key in cfg_dict.keys():
        if isinstance(cfg_dict[key], dict):
            if "sweep_grid" in cfg_dict[key]:
                final_list.append(carry_tup + (key,))
            else:
                final_list.extend(
                    find_sweep_grids(cfg_dict[key], carry_tup=carry_tup + (key,))
                )
    return final_list


def find_sg_values(cfg_dict, sg_key_tups):
    sg_values = []
    for tup in sg_key_tups:
        nested_dict = cfg_dict
        for key in tup:
            nested_dict = nested_dict[key]
        sg_values.append(list(nested_dict["sweep_grid"].values()))
    return sg_values


def binary_classifier_performance(scores, y_hats, print_and_plot=False):
    frac_pos = np.sum(y_hats) / len(y_hats)

    roc_auc = roc_auc_score(y_hats, scores)
    avg_prc = average_precision_score(y_hats, scores)

    fpr, tpr, _ = roc_curve(y_hats, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_hats, scores)

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.nan_to_num(2 * recall * precision / (recall + precision))
    best_treshold = pr_thresholds[np.argmax(f1_scores)]

    best_f1_score = np.max(f1_scores)

    preds = np.float32(scores > best_treshold)
    cm = confusion_matrix(y_hats, preds)

    cm_05 = confusion_matrix(y_hats, np.round(scores))
    tn, fp, fn, tp = cm_05.ravel()

    acc_05 = accuracy_score(y_hats, np.round(scores))

    if print_and_plot:
        print("Imbalance: %.2f" % frac_pos)
        print("ROC AUC Score: %.2f" % roc_auc)
        print("Average Precision: %.2f" % avg_prc)
        print("Best threshold: ", best_treshold)
        print("Best F1-Score: ", best_f1_score)
        print("Confusion matrix:")
        print(cm_05)
        print(f"Accuracy: {acc_05}")
        print(f"TPR: {tp / (tp + fn)}")

        plt.plot(fpr, tpr)
        plt.show()
        plt.plot(recall, precision)
        plt.show()

    return frac_pos, roc_auc, avg_prc, best_f1_score, acc_05

import os
import pickle
import tempfile
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def get_current_time_stamp_minutes():
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d_%H_%M")
    return current_time


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


def tensor_to_array(tensor):
    return tensor.cpu().detach().numpy()


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


def save_file(
    file,
    filename,
    experiment,
    project,
    entity,
    home_dir,
    mode="online",
    config=None,
    local_path=None,
):
    if mode == "online":
        with tempfile.TemporaryDirectory() as tmpdir:
            run = wandb.init(
                mode=mode,
                project=project,
                entity=entity,
                dir=home_dir,
                group=experiment,
                config=config,
                name=f"SAVING_RUN:{filename}",
            )
            temp_file_path = os.path.join(tmpdir, filename)
            with open(temp_file_path, "wb") as f:
                pickle.dump(file, f)
            run.save(temp_file_path, base_path=tmpdir)
            run.finish()
    if local_path is not None:
        save_pickle(file, os.path.join(local_path, experiment, filename))
    if mode == "offline" and local_path is None:
        raise ValueError("No valid save mode specified.")


def wandb_loader(filename, run_path):
    data = None
    with tempfile.TemporaryDirectory() as tmpdir:
        wandb.restore(name=filename, run_path=run_path, root=tmpdir)
        with open(os.path.join(tmpdir, filename), "rb") as f:
            data = pickle.load(f)
    return data

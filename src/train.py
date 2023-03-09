import logging
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader

from src.models import GlobalAveragePooling, NeuralAdditiveModel, WeightedMultilabel
from src.neonatal_utils import NeoNatal, read_data
from src.utils import (
    MultiDatasetBalancedSampler,
    create_sweep_cfgs,
    save_pickle,
    tensor_to_array,
)


log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = "neonatal"
HOME_PATH = None
WANDB_ENTITY = None


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)


def train_classifier(classifier, data_loader, criterion, optimizer):
    """
    Train the classifier (one epoch).
    """

    classifier.train()

    num_samples = 0
    agg_los = 0.0
    concat_ys = np.array([])
    concat_scores = np.array([])
    for itr, batch in enumerate(data_loader):
        xs, ys = batch
        batch_size = ys.shape[0]

        if isinstance(classifier, NeuralAdditiveModel):
            xs = [sig.to(DEVICE) for sig in xs]  # NAM
        elif isinstance(classifier, GlobalAveragePooling):
            xs = xs.to(DEVICE)  # Standard
        else:
            raise ValueError

        ys = ys.to(DEVICE)

        y_hats = classifier.forward(xs)
        loss = criterion(y_hats, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_samples += batch_size
        agg_los += loss.item() * batch_size
        concat_ys = np.concatenate([concat_ys, tensor_to_array(ys).flatten()])
        concat_scores = np.concatenate(
            [concat_scores, tensor_to_array(y_hats).flatten()]
        )

    return (agg_los / num_samples, concat_ys, concat_scores)


def test_classifier(classifier, data_loader, criterion):
    """
    Evalutate the classifier.
    """

    classifier.eval()

    num_samples = 0
    agg_los = 0.0
    concat_ys = np.array([])
    concat_scores = np.array([])
    for itr, batch in enumerate(data_loader):
        xs, ys = batch
        batch_size = ys.shape[0]

        if isinstance(classifier, NeuralAdditiveModel):
            xs = [sig.to(DEVICE) for sig in xs]  # GAM
        elif isinstance(classifier, GlobalAveragePooling):
            xs = xs.to(DEVICE)  # Standard
        else:
            raise ValueError

        ys = ys.to(DEVICE)

        y_hats = classifier.forward(xs)
        loss = criterion(y_hats, ys)

        num_samples += batch_size
        agg_los += loss.item() * batch_size
        concat_ys = np.concatenate([concat_ys, tensor_to_array(ys).flatten()])
        concat_scores = np.concatenate(
            [concat_scores, tensor_to_array(y_hats).flatten()]
        )

    return (agg_los / num_samples, concat_ys, concat_scores)


# Assumes single batch test_data
def get_full_performance_and_logits(dataset, classifier):
    """
    Get all additive contributions (logits) and perforamce measures
    for a given classifier and dataset.

    Should only be used when the test data is processed as a single batch.
    """

    classifier = classifier.to(DEVICE)
    classifier.eval()

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=len(dataset)):
            xs, ys = batch

            if isinstance(classifier, NeuralAdditiveModel):
                xs = [sig.to(DEVICE) for sig in xs]  # NAM
            else:
                xs = xs.to(DEVICE)  # Standard
            ys = ys.to(DEVICE)

            with torch.no_grad():
                y_hats = classifier.forward(xs)

            if isinstance(classifier, NeuralAdditiveModel):
                logits, bias = classifier.compute_logits(xs)
                logits = [tensor_to_array(mod).flatten() for mod in logits]
                bias = bias.item()  # will always be the same
            else:
                logits, bias = None, None

            labels = tensor_to_array(ys).flatten()
            scores = tensor_to_array(y_hats).flatten()

    return labels, scores, logits, bias


def init_neonatal_dataset(pat_id, signal_dict, cfg):
    "Initialize the data set of a single patient with the parameters specified in cfg."

    return NeoNatal(
        pat_id=pat_id,
        signal_dict=signal_dict,
        dataset_mode=cfg.dataset.dataset_mode,
        signal_types=cfg.dataset.signal_types,
        adverse_events=cfg.dataset.adverse_events,
        cutter_events=cfg.dataset.cutter_events,
        time_window=cfg.dataset.time_window,
        lag=cfg.dataset.lag,
        away=cfg.dataset.away,
    )


def init_classifier(cfg):
    """
    Initialize the classification model based on the type specified in cfg.
    """

    if cfg.network.self == "nam":
        classifier_lst = [
            GlobalAveragePooling(
                in_channel=in_channels,
                hidden_channel=hidden_channels,
                kernel_size=kernel_size,
                dilation=1,
            )
            for in_channels, hidden_channels, kernel_size in zip(
                cfg.network.in_channels,
                cfg.network.hidden_channels,
                cfg.network.kernel_size,
            )
        ]

        classifier = NeuralAdditiveModel(classifier_lst)
    elif cfg.network.self == "gap":
        classifier = GlobalAveragePooling(
            cfg.network.in_channels,
            hidden_channel=cfg.network.hidden_channels,
            kernel_size=cfg.network.kernel_size,
        )
    else:
        raise ValueError

    return classifier


def run_epoch(
    classifier,
    optimizer,
    criterion,
    train_loader,
    test_loader,
):
    """
    Run a single training and testing epoch and record performance measures.
    """

    test_loss, test_labs, test_scores = test_classifier(
        classifier, test_loader, criterion
    )
    test_metrics = {
        "agg_test_loss": test_loss,
        "agg_test_acc": accuracy_score(np.round(test_scores), test_labs),
        "agg_test_auc": roc_auc_score(test_labs, test_scores),
        "agg_test_avp": average_precision_score(test_labs, test_scores),
    }
    train_loss, train_labs, train_scores = train_classifier(
        classifier,
        train_loader,
        criterion,
        optimizer,
    )
    train_metrics = {
        "agg_train_loss": train_loss,
        "agg_train_acc": accuracy_score(np.round(train_scores), train_labs),
        "agg_train_auc": roc_auc_score(train_labs, train_scores),
        "agg_train_avp": average_precision_score(train_labs, train_scores),
    }
    log.info(str(test_metrics))
    log.info(str(train_metrics))


# Test data can also be val data here!
def train_model_and_evaluate_performance(cfg, train_dataset, test_dataset):
    """
    Train the model on the training set for a given set of hyperparameters and evaluate performance on the test set.
    """

    train_loader = DataLoader(
        train_dataset,
        cfg.optimizer.train_batch_size,
        sampler=MultiDatasetBalancedSampler(train_dataset, replacement=False),
        pin_memory=True,
    )
    test_loader = DataLoader(test_dataset, len(test_dataset), pin_memory=True)

    classifier = init_classifier(cfg)
    classifier = classifier.to(DEVICE)
    weights = torch.Tensor([cfg.optimizer.loss_weight, 1.0])
    weights = weights / weights.sum()
    weights = weights.to(DEVICE)
    criterion = WeightedMultilabel(weights)
    optimizer = optim.Adam(
        classifier.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )

    for _epoch in range(cfg.optimizer.epochs):
        run_epoch(
            classifier=classifier,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
        )

    _, final_test_labs, final_test_scores = test_classifier(
        classifier, test_loader, criterion
    )

    return classifier.state_dict(), final_test_labs, final_test_scores


def inner_cv_loop(dict_cfg, ind_training_datasets):
    """
    Run the inner loop in the nested leave-one-out cross validation.
    """

    results_dict = {}
    for dat_idx, val_data in enumerate(ind_training_datasets):
        val_train_dataset_list = [
            ind_training_datasets[jdx]
            for jdx, _dat in enumerate(ind_training_datasets)
            if jdx != dat_idx
        ]
        val_train_dataset = ConcatDataset(val_train_dataset_list)

        (
            _model_dict,
            final_test_labs,
            final_test_scores,
        ) = train_model_and_evaluate_performance(
            dict_cfg, val_train_dataset, val_data, log_to_wandb=False
        )

        results_dict[str(dat_idx)] = {
            "ys": final_test_labs,
            "scores": final_test_scores,
        }

    return np.mean(
        [roc_auc_score(dic["ys"], dic["scores"]) for dic in results_dict.values()]
    )


def nested_leave_one_out(cfg):
    """
    Run leave-one-out cross validation.
    """

    log.info("Create Datasets")
    single_datasets = {}
    for pat_id in cfg.dataset.ids:
        log.info(f"Processing id {pat_id}")
        signal_dict = read_data(pat_id, cfg.meta.data_path)
        single_datasets[pat_id] = init_neonatal_dataset(pat_id, signal_dict, cfg)
    log.info("Done.")

    results_dict = {}
    models_dict = {}

    for pat_id in cfg.dataset.ids:
        # Remove test patient
        ind_train_datasets = [
            single_datasets[pat_jd] for pat_jd in cfg.dataset.ids if pat_jd != pat_id
        ]

        sweep_cfgs = create_sweep_cfgs(cfg)
        log.debug(sweep_cfgs)

        # Nested CV
        if len(sweep_cfgs) > 1:
            log.info("Start inner loop!")
            best_cfg = None
            best_performance = 0.0
            for network_parameters in sweep_cfgs:
                log.info(network_parameters)
                average_performance = inner_cv_loop(
                    network_parameters, ind_train_datasets
                )
                log.info(average_performance)
                if average_performance > best_performance:
                    best_performance = average_performance
                    best_cfg = network_parameters

            log.info("START Best network parameters and performance")
            log.info(best_cfg)
            log.info(best_performance)
            log.info("END Best network parameters and performance")
        else:
            log.info("No grids detected. Running without nested loop.")
            best_cfg = sweep_cfgs[0]

        train_dataset = ConcatDataset(ind_train_datasets)
        test_dataset = single_datasets[pat_id]
        log.info(len(train_dataset))
        log.info(len(test_dataset))

        (
            state_dic,
            final_test_labs,
            final_test_scores,
        ) = train_model_and_evaluate_performance(best_cfg, train_dataset, test_dataset)

        results_dict[pat_id] = {"ys": final_test_labs, "scores": final_test_scores}
        models_dict[pat_id] = state_dic

    log.info("Saving!")
    save_pickle(
        obj=results_dict,
        file_path=os.path.join(
            best_cfg.meta.result_path, f"{best_cfg.meta.tag}_results.pkl"
        ),
    )
    save_pickle(
        obj=models_dict,
        file_path=os.path.join(
            best_cfg.meta.result_path, f"{best_cfg.meta.tag}_models.pkl"
        ),
    )
    save_pickle(
        obj=best_cfg,
        file_path=os.path.join(
            best_cfg.meta.result_path, f"{best_cfg.meta.tag}_config.pkl"
        ),
    )

    return np.mean(
        [roc_auc_score(dic["ys"], dic["scores"]) for dic in results_dict.values()]
    )


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="neonatal_config",
)
def main(cfg):
    """
    Wrap leave-one-out cross validation and log some information.
    """

    log.info("Cuda is available: " + str(torch.cuda.is_available()))

    log.info(OmegaConf.to_yaml(cfg))
    log.info("Config correct? Abort within 10s if not.")
    time.sleep(10)
    log.info("Start run.")

    if cfg.meta.seed == "none":
        random_random_seed = np.random.randint(2**32)
        set_seed(random_random_seed)
    else:
        set_seed(cfg.meta.seed)

    average_test_auc = nested_leave_one_out(cfg)
    log.info({"average_test_auc": average_test_auc})


if __name__ == "__main__":
    main()

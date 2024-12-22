import logging
import os
import time

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn, optim
from torch.utils.data import ConcatDataset, DataLoader

from src.misc_utils import save_file, tensor_to_array
from src.modules import (
    GlobalAveragePooling,
    LogisticRegression,
    MLPClassifier,
    MLPCombiner,
    NeuralAdditiveModel,
)
from src.neonatal_utils import MultiDatasetBalancedSampler, NeoNatal, read_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log = logging.getLogger(__name__)


def set_seed(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_val)


def init_neonatal_dataset(pat_id, cfg):
    if cfg.meta.pickle_path is None:
        signal_dict = read_data(pat_id, cfg.meta.data_path)
    else:
        signal_dict = pd.read_pickle(
            os.path.join(cfg.meta.pickle_path, pat_id + ".pkl")
        )
    return NeoNatal(
        pat_id,
        signal_dict,
        dataset_mode=cfg.dataset.dataset_mode,
        signal_types=cfg.dataset.signal_types,
        adverse_events=cfg.dataset.adverse_events,
        cutter_events=cfg.dataset.cutter_events,
        time_window=cfg.dataset.time_window,
        lag=cfg.dataset.lag,
        away=cfg.dataset.away,
    )


def init_classifier(cfg):
    if cfg.network.self == "nam":
        classifier_lst = [
            GlobalAveragePooling(
                in_channel=in_channels,
                hidden_channel=hidden_channels,
                kernel_size=kernel_size,
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
    elif cfg.network.self == "mlp":
        classifier = MLPClassifier(
            meta_size=cfg.network.feature_dim,
            intermediate_size=cfg.network.hidden_dim,
        )
    elif cfg.network.self == "log_reg":
        classifier = LogisticRegression(cfg.network.feature_dim)
    elif cfg.network.self == "mlp_combiner":
        classifier_lst = [
            GlobalAveragePooling(
                in_channel=in_channels,
                hidden_channel=hidden_channels,
                kernel_size=kernel_size,
            )
            for in_channels, hidden_channels, kernel_size in zip(
                cfg.network.in_channels,
                cfg.network.hidden_channels,
                cfg.network.kernel_size,
            )
        ]
        mlp_classifier = MLPClassifier(
            sum(cfg.network.hidden_channels), cfg.network.mlp_hidden_dim
        )
        classifier = MLPCombiner(classifier_lst, mlp_classifier)
    else:
        raise ValueError

    return classifier


def LBFGS_closure(classifier, xs, ys, criterion, optimizer, l2_lambda):
    optimizer.zero_grad()
    y_hats = classifier.forward(xs)
    loss = criterion(y_hats, ys)

    l2_reg = torch.tensor(0.0, requires_grad=True)
    for param in classifier.parameters():
        l2_reg = l2_reg + torch.norm(param, 2)
    loss = loss + l2_lambda * l2_reg

    loss.backward()
    return loss


def train_classifier(
    classifier,
    data_loader,
    criterion,
    optimizer,
    mean_feats=None,
    std_feats=None,
    l2_lambda=None,  # only for L-BFGS
):
    classifier.train()

    num_samples = 0
    agg_los = 0.0
    concat_ys = np.array([])
    concat_scores = np.array([])
    for batch in data_loader:
        xs, ys = batch
        batch_size = ys.shape[0]

        if mean_feats is not None and std_feats is not None:
            xs = (xs - mean_feats) / std_feats

        if isinstance(xs, list):
            xs = [sig.to(device) for sig in xs]
        else:
            xs = xs.to(device)

        ys = ys.to(device)

        y_hats = classifier.forward(xs)
        loss = criterion(y_hats, ys)  # just compute twice for LBFGS

        if isinstance(optimizer, optim.LBFGS):
            optimizer.step(
                lambda: LBFGS_closure(
                    classifier, xs, ys, criterion, optimizer, l2_lambda
                )
            )
        else:
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


def test_classifier(
    classifier,
    data_loader,
    criterion,
    mean_feats=None,
    std_feats=None,
):
    classifier.eval()

    num_samples = 0
    agg_los = 0.0
    concat_ys = np.array([])
    concat_scores = np.array([])
    for batch in data_loader:
        xs, ys = batch
        batch_size = ys.shape[0]

        if mean_feats is not None and std_feats is not None:
            xs = (xs - mean_feats) / std_feats

        if isinstance(xs, list):
            xs = [sig.to(device) for sig in xs]
        else:
            xs = xs.to(device)

        ys = ys.to(device)

        y_hats = classifier.forward(xs)
        loss = criterion(y_hats, ys)

        num_samples += batch_size
        agg_los += loss.item() * batch_size
        concat_ys = np.concatenate([concat_ys, tensor_to_array(ys).flatten()])
        concat_scores = np.concatenate(
            [concat_scores, tensor_to_array(y_hats).flatten()]
        )

    return (agg_los / num_samples, concat_ys, concat_scores)


def class_activation(classifier, data_loader):
    classifier.eval()

    # assumes that data is processed in  a single batch
    for batch in data_loader:
        xs, ys = batch
        xs = xs.to(device)

        with torch.no_grad():
            y_hats = classifier.forward(xs)
            class_acts = classifier.class_act(xs)

        concat_xs = tensor_to_array(xs)
        concat_scores = tensor_to_array(y_hats).flatten()
        concat_class_acts = tensor_to_array(class_acts)
        concat_ys = ys.numpy().flatten()

    return {
        "xs": concat_xs,
        "scores": concat_scores,
        "ys": concat_ys,
        "class_activations": concat_class_acts,
    }


# Assumes single batch test_data
def get_full_performance_and_logits(dataset, classifier):
    classifier = classifier.to(device)
    classifier.eval()

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=len(dataset)):
            xs, ys = batch

            if isinstance(xs, list):
                xs = [sig.to(device) for sig in xs]
            else:
                xs = xs.to(device)

            ys = ys.to(device)

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


# Test data can also be val data here!
def train_model_and_evaluate_performance(
    cfg,
    train_dataset,
    test_dataset,
    log_to_wandb=True,
    mean_feats=None,
    std_feats=None,
    l2_lambda=None,  # only for LBFGS
):
    train_loader = DataLoader(
        train_dataset,
        cfg.optimizer.train_batch_size,
        sampler=MultiDatasetBalancedSampler(train_dataset, replacement=False),
        pin_memory=True,
    )
    test_loader = DataLoader(test_dataset, len(test_dataset), pin_memory=True)

    classifier = init_classifier(cfg)
    classifier = classifier.to(device)
    criterion = nn.BCELoss()
    if cfg.optimizer.self == "adam_w":
        optimizer = optim.AdamW(
            classifier.parameters(),
            lr=cfg.optimizer.learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.self == "LBFGS":
        optimizer = optim.LBFGS(
            classifier.parameters(),
            lr=cfg.optimizer.learning_rate,
            history_size=cfg.optimizer.history_size,
        )
        l2_lambda = cfg.optimizer.l2_lambda
    else:
        raise ValueError

    for _epoch in range(cfg.optimizer.epochs):
        test_loss, test_labs, test_scores = test_classifier(
            classifier,
            test_loader,
            criterion,
            mean_feats,
            std_feats,
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
            mean_feats,
            std_feats,
            l2_lambda,
        )
        train_metrics = {
            "agg_train_loss": train_loss,
            "agg_train_acc": accuracy_score(np.round(train_scores), train_labs),
            "agg_train_auc": roc_auc_score(train_labs, train_scores),
            "agg_train_avp": average_precision_score(train_labs, train_scores),
        }
        if log_to_wandb:
            wandb.log(test_metrics, commit=False)
            wandb.log(train_metrics)

    _, final_test_labs, final_test_scores = test_classifier(
        classifier, test_loader, criterion, mean_feats, std_feats
    )

    return classifier.state_dict(), final_test_labs, final_test_scores


def nested_leave_one_out(cfg):
    log.info("Create Datasets")
    single_datasets = {}

    for pat_id in cfg.dataset.ids:
        log.info(f"Processing id {pat_id}")
        single_datasets[pat_id] = init_neonatal_dataset(pat_id, cfg)
    log.info("Done.")

    # save per patient results
    results_dict = {}
    models_dict = {}

    for pat_id in cfg.dataset.ids:
        run = wandb.init(
            mode=cfg.meta.mode,
            project=cfg.meta.wandb_project,
            entity=cfg.meta.wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            dir=cfg.meta.wandb_home,
            group=cfg.meta.experiment,
            name=f"{cfg.meta.tag}_{pat_id}",
        )

        start = time.time()

        # remove test pat
        ind_train_datasets = [
            single_datasets[pat_jd] for pat_jd in cfg.dataset.ids if pat_jd != pat_id
        ]

        train_dataset = ConcatDataset(ind_train_datasets)
        test_dataset = single_datasets[pat_id]

        # global standardization for features
        mean_feats = None
        std_feats = None
        if cfg.dataset.dataset_mode == "features":
            feats = torch.stack([sig for sig, _ in train_dataset])
            mean_feats = torch.mean(feats, dim=0)
            std_feats = torch.std(feats, dim=0)

        run.summary["train_data_size"] = len(train_dataset)
        run.summary["test_data_size"] = len(test_dataset)

        (
            state_dic,
            final_test_labs,
            final_test_scores,
        ) = train_model_and_evaluate_performance(
            cfg,
            train_dataset,
            test_dataset,
            mean_feats=mean_feats,
            std_feats=std_feats,
        )

        results_dict[pat_id] = {"ys": final_test_labs, "scores": final_test_scores}
        models_dict[pat_id] = state_dic

        log.info(time.time() - start)

        run.finish()

    log.info("Saving.")
    save_file(
        file=results_dict,
        filename=f"{cfg.meta.tag}_results.pkl",
        experiment=cfg.meta.experiment,
        project=cfg.meta.wandb_project,
        entity=cfg.meta.wandb_entity,
        home_dir=cfg.meta.wandb_home,
        mode=cfg.meta.mode,
        local_path=cfg.meta.result_path,
    )
    save_file(
        file=models_dict,
        filename=f"{cfg.meta.tag}_models.pkl",
        experiment=cfg.meta.experiment,
        project=cfg.meta.wandb_project,
        entity=cfg.meta.wandb_entity,
        home_dir=cfg.meta.wandb_home,
        mode=cfg.meta.mode,
        local_path=cfg.meta.result_path,
    )
    save_file(
        file=cfg,
        filename=f"{cfg.meta.tag}_config.pkl",
        experiment=cfg.meta.experiment,
        project=cfg.meta.wandb_project,
        entity=cfg.meta.wandb_entity,
        home_dir=cfg.meta.wandb_home,
        mode=cfg.meta.mode,
        local_path=cfg.meta.result_path,
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
    log.info("Cuda is available: " + str(torch.cuda.is_available()))

    log.info(OmegaConf.to_yaml(cfg))
    log.info("Config correct? Abort if not.")
    time.sleep(5)
    log.info("Start run.")

    assert cfg.meta.experiment is not None
    assert cfg.meta.tag is not None

    if cfg.meta.seed is None:
        random_random_seed = np.random.randint(2**32)
        set_seed(random_random_seed)
    else:
        set_seed(cfg.meta.seed)

    average_test_auc = nested_leave_one_out(cfg)

    run = wandb.init(
        mode=cfg.meta.mode,
        project=cfg.meta.wandb_project,
        entity=cfg.meta.wandb_entity,
        dir=cfg.meta.wandb_home,
        group=cfg.meta.experiment,
        name=f"{cfg.meta.tag}_summary",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    run.log({"average_test_auc": average_test_auc})
    if cfg.meta.seed == "none":
        run.log({"random_random_seed": random_random_seed})

    run.finish()


if __name__ == "__main__":
    main()

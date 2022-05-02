"""Training ProtMIMOOracle for Stability data."""

import os
import argparse
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from pprint import pprint
from tape.datasets import LMDBDataset

from model import ProtMIMOOracle
from data_utils import (
    DatasetType,
    create_batched_train_data,
    create_batched_test_data,
    create_plot,
    train_and_evaluate,
    train_and_evaluate_ensemble,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Determine model type.")
parser.add_argument("convolutional", type=bool, nargs="?", default=False)
args = parser.parse_args()
USE_CONV_MODEL = args.convolutional


STABILITY_SEQ_LEN = 50
STABILITY_AMINO_ACID_VOCABULARY = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    ".",
]
STABILITY_ALPHABET = {}
for i, aa in enumerate(STABILITY_AMINO_ACID_VOCABULARY):
    STABILITY_ALPHABET[aa] = i


def stability_dataset_to_df(in_name):
    dataset = LMDBDataset(in_name)
    df = pd.DataFrame(list(dataset)[:])
    df["stability"] = df.stability_score.apply(lambda x: x[0])
    df["id_str"] = df.id.apply(lambda x: x.decode("utf-8"))
    return df


def get_parent_to_parent_stability():
    try:
        stability_train_df = stability_dataset_to_df("stability/stability_train.lmdb")
        stability_val_df = stability_dataset_to_df("stability/stability_valid.lmdb")
        stability_test_df = stability_dataset_to_df("stability/stability_test.lmdb")
    except FileNotFoundError:
        os.system(
            "wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz"
        )
        os.system("tar xzf stability.tar.gz")
        stability_train_df = stability_dataset_to_df("stability/stability_train.lmdb")
        stability_val_df = stability_dataset_to_df("stability/stability_valid.lmdb")
        stability_test_df = stability_dataset_to_df("stability/stability_test.lmdb")

    parent_to_parent_stability = {}
    for parent in (
        set(stability_train_df.parent.values)
        | set(stability_val_df.parent.values)
        | set(stability_test_df.parent.values)
    ):
        stabilities = stability_train_df[
            stability_train_df["id_str"] == parent.decode("utf-8") + ".pdb"
        ].stability.values
        if len(stabilities) == 0:
            stabilities = stability_train_df[
                stability_train_df["id_str"] == parent.decode("utf-8")
            ].stability.values
            if len(stabilities) == 0:
                parent_to_parent_stability[parent] = None
            else:
                parent_to_parent_stability[parent] = stabilities[0]
        else:
            parent_to_parent_stability[parent] = stabilities[0]

    return parent_to_parent_stability


def topology_to_index(top):
    topology_to_ind = {"HHH": 0, "HEEH": 1, "EHEE": 3, "EEHEE": 4}
    top = top.decode("utf-8")
    if top in topology_to_ind.keys():
        return topology_to_ind[top]
    else:
        return 2


def create_stability_df(parent_to_parent_stability, dataset_type):
    if dataset_type == DatasetType.Train:
        stability_df = stability_dataset_to_df("stability/stability_train.lmdb")
    elif dataset_type == DatasetType.Val:
        stability_df = stability_dataset_to_df("stability/stability_valid.lmdb")
    elif dataset_type == DatasetType.Test:
        stability_df = stability_dataset_to_df("stability/stability_test.lmdb")
    stability_df["parent_stability"] = stability_df.parent.apply(
        lambda x: parent_to_parent_stability[x]
    )
    stability_df["topology_ind"] = stability_df.topology.apply(
        lambda x: topology_to_index(x)
    )

    return stability_df


def get_stability_dfs():
    parent_to_parent_stability = get_parent_to_parent_stability()
    train_df = create_stability_df(
        parent_to_parent_stability=parent_to_parent_stability,
        dataset_type=DatasetType.Train,
    )
    val_df = create_stability_df(
        parent_to_parent_stability=parent_to_parent_stability,
        dataset_type=DatasetType.Val,
    )
    test_df = create_stability_df(
        parent_to_parent_stability=parent_to_parent_stability,
        dataset_type=DatasetType.Test,
    )
    return train_df, val_df, test_df


train_df, val_df, test_df = get_stability_dfs()


def compute_stability_scalar_metrics(targets, preds, prefix=None, test_df=test_df):
    scalar_metrics = {}
    if prefix:
        prefix = f"{prefix}_"
    else:
        prefix = ""
    test_df["targets"] = targets
    test_df["preds"] = preds

    base_spearmanr = spearmanr(targets, preds).correlation
    base_mse = mean_squared_error(targets, preds)
    scalar_metrics[f"{prefix}spearmanr"] = base_spearmanr
    scalar_metrics[f"{prefix}mse"] = base_mse

    parent_to_pred_parent_stability = {}
    for parent in set(test_df.parent.values):
        pred_parent_stabilities = test_df[
            test_df["id_str"] == parent.decode("utf-8") + ".pdb"
        ].preds.values
        if len(pred_parent_stabilities) == 0:
            pred_parent_stabilities = test_df[
                test_df["id_str"] == parent.decode("utf-8")
            ].preds.values
            if len(pred_parent_stabilities) == 0:
                parent_to_pred_parent_stability[parent] = None
            else:
                parent_to_pred_parent_stability[parent] = pred_parent_stabilities[0]
        else:
            parent_to_pred_parent_stability[parent] = pred_parent_stabilities[0]
    test_df["pred_parent_stability"] = test_df.parent.apply(
        lambda x: parent_to_pred_parent_stability[x]
    )

    correct_direction = 0
    for test_stability, pred_stability, parent_stability, pred_parent_stability in zip(
        targets,
        preds,
        test_df.parent_stability.values,
        test_df.pred_parent_stability.values,
    ):
        if parent_stability is not None:
            if (
                test_stability >= parent_stability
                and pred_stability >= pred_parent_stability
            ) or (
                test_stability <= parent_stability
                and pred_stability <= pred_parent_stability
            ):
                correct_direction += 1
    base_accuracy = correct_direction / len(
        np.where(test_df.parent_stability.values != None)[0]
    )
    scalar_metrics[f"{prefix}accuracy"] = base_accuracy

    test_df["topology_str"] = test_df.topology.apply(lambda x: x.decode("utf-8"))
    topologies = [
        ("EEHEE", "BBABB"),
        ("EHEE", "BABB"),
        ("HEEH", "ABBA"),
        ("HHH", "AAA"),
    ]
    for topology_pair in topologies:
        topology, topology_name = topology_pair
        topology_test_df = test_df[test_df["topology_str"] == topology]

        topology_spearmanr = spearmanr(
            topology_test_df.targets.values, topology_test_df.preds.values
        ).correlation
        topology_mse = mean_squared_error(
            topology_test_df.targets.values, topology_test_df.preds.values
        )
        scalar_metrics[f"{prefix}{topology_name}_spearmanr"] = topology_spearmanr
        scalar_metrics[f"{prefix}{topology_name}_mse"] = topology_mse

        topology_correct_direction = 0
        for (
            test_stability,
            pred_stability,
            parent_stability,
            pred_parent_stability,
        ) in zip(
            topology_test_df.targets.values,
            topology_test_df.stability.values,
            topology_test_df.parent_stability.values,
            topology_test_df.pred_parent_stability.values,
        ):
            if parent_stability is not None:
                if (
                    test_stability >= parent_stability
                    and pred_stability >= pred_parent_stability
                ) or (
                    test_stability <= parent_stability
                    and pred_stability <= pred_parent_stability
                ):
                    topology_correct_direction += 1
        topology_accuracy = topology_correct_direction / len(
            np.where(topology_test_df.parent_stability.values != None)[0]
        )
        scalar_metrics[f"{prefix}{topology_name}_accuracy"] = topology_accuracy

    for metric_key in scalar_metrics.keys():
        scalar_metrics[metric_key] = float(scalar_metrics[metric_key])

    return scalar_metrics


def get_stability_metrics(
    targets,
    preds,
    preds_by_input,
    num_inputs,
    loss_fn,
    plot=False,
    test_df=test_df,
    ensemble=False,
):
    metrics = {}
    test_loss = loss_fn(torch.tensor(targets), torch.tensor(preds)).item()
    metrics["test_loss"] = test_loss
    metrics.update(
        compute_stability_scalar_metrics(
            targets=targets, preds=preds, prefix=None, test_df=test_df
        )
    )
    for i in range(num_inputs):
        metrics[f"model_{i}_test_loss"] = loss_fn(
            torch.tensor(targets), torch.tensor(preds_by_input[f"model_{i}"])
        ).item()
        metrics.update(
            compute_stability_scalar_metrics(
                targets=targets, preds=preds, prefix=f"model_{i}", test_df=test_df
            )
        )
        for j in range(i + 1, num_inputs):
            metrics[f"model_{i}_{j}_residual_correlation"] = pearsonr(
                preds_by_input[f"model_{i}"], preds_by_input[f"model_{j}"]
            )[0]

    if plot
        if ensemble:
            title = f"Ensemble of {num_inputs} CNN Models"
            path = f'stability_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/ensemble_of_{num_inputs}_models.jpg'
        else:
            title = f"MIMO CNN Model with {num_inputs} Inputs and Outputs"
            path = f'stability_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/mimo_with_{num_inputs}_inputs.jpg'
        create_plot(
            targets=targets,
            preds=preds,
            title=title,
            path=path,
            feature_name="Stability",
        )

        for i in range(num_inputs):
            if ensemble:
                title = f"CNN Model {i + 1} of Ensemble of {num_inputs} Models"
                path = f'stability_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/model_{i+1}_of_ensemble_of_{num_inputs}_models.jpg'
            else:
                title = f"Output {i + 1} of MIMO CNN Model with {num_inputs} Inputs and Outputs"
                path = f'stability_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/output_{i+1}_of_mimo_with_{num_inputs}_inputs.jpg'
            create_plot(
                targets,
                preds_by_input[f"model_{i}"],
                title=title,
                path=path,
                feature_name="Stability",
            )

    return metrics


# MIMO Training/Evaluation
def train_stability_mimo_model(
    hidden_dim,
    feed_forward_kwargs,
    conv_kwargs,
    num_inputs=3,
    bs=32,
    lr=0.001,
    num_epochs=100,
    patience=10,
    plot=False,
):
    training_data = create_batched_train_data(
        train_df=train_df, num_inputs=num_inputs, bs=bs, feature_name="stability"
    )
    val_data = create_batched_train_data(
        train_df=val_df, num_inputs=num_inputs, bs=bs, feature_name="stability"
    )
    test_data = create_batched_test_data(
        test_df=test_df, num_inputs=num_inputs, bs=bs, feature_name="stability"
    )
    data = {
        "training_data": training_data,
        "val_data": val_data,
        "test_data": test_data,
    }

    if USE_CONV_MODEL:
        model = ProtMIMOOracle(
            alphabet=STABILITY_ALPHABET,
            max_len=STABILITY_SEQ_LEN,
            num_inputs=num_inputs,
            hidden_dim=hidden_dim,
            convolutional=True,
            conv_kwargs=conv_kwargs,
        )
    else:
        model = ProtMIMOOracle(
            alphabet=STABILITY_ALPHABET,
            max_len=STABILITY_SEQ_LEN,
            num_inputs=num_inputs,
            hidden_dim=hidden_dim,
            feed_forward_kwargs=feed_forward_kwargs,
        )
    model.to(DEVICE)

    best_model, test_metrics = train_and_evaluate(
        model=model,
        data=data,
        metrics_fn=get_stability_metrics,
        plot=plot,
        num_inputs=num_inputs,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
    )
    return test_metrics


# Standard Ensemble
def train_stability_ensemble_models(
    hidden_dim,
    feed_forward_kwargs,
    conv_kwargs,
    num_inputs=3,
    bs=32,
    lr=0.001,
    num_epochs=100,
    patience=10,
    plot=False,
):
    data = []
    for i in range(num_inputs):
        ensemble_training_data = create_batched_train_data(
            train_df=train_df,
            num_inputs=1,
            bs=bs,
            feature_name="stability",
            ensemble_model_num=i,
        )
        ensemble_val_data = create_batched_train_data(
            train_df=val_df,
            num_inputs=1,
            bs=bs,
            feature_name="stability",
            ensemble_model_num=i,
        )
        ensemble_test_data = create_batched_test_data(
            test_df=test_df, num_inputs=1, bs=bs, feature_name="stability"
        )
        data.append(
            {
                "training_data": ensemble_training_data,
                "val_data": ensemble_val_data,
                "test_data": ensemble_test_data,
            }
        )
    models = [
        (
            ProtMIMOOracle(
                alphabet=STABILITY_ALPHABET,
                max_len=STABILITY_SEQ_LEN,
                num_inputs=1,
                hidden_dim=hidden_dim,
                convolutional=True,
                conv_kwargs=conv_kwargs,
            )
            if USE_CONV_MODEL
            else ProtMIMOOracle(
                alphabet=STABILITY_ALPHABET,
                max_len=STABILITY_SEQ_LEN,
                num_inputs=1,
                hidden_dim=hidden_dim,
                feed_forward_kwargs=feed_forward_kwargs,
            )
        ).to(DEVICE)
        for _ in range(num_inputs)
    ]

    best_models, test_metrics = train_and_evaluate_ensemble(
        models=models,
        data=data,
        metrics_fn=get_stability_metrics,
        plot=plot,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
    )
    return test_metrics


if __name__ == "__main__":
    hidden_dim = 512
    num_inputs = 3
    bs = 32
    lr = 0.0001
    num_epochs = 100
    patience = 10
    for num_layers in range(1, 11):
        feed_forward_kwargs = {
            "hidden_dims": [256] * num_layers,
        }
        conv_kwargs = {
            "channels": [64] * num_layers,
            "kernel_sizes": [5] * num_layers,
            "pooling_dims": [0] * num_layers,
        }
        parameters = {
            "hidden_dim": hidden_dim,
            "num_inputs": num_inputs,
            "bs": bs,
            "lr": lr,
            "num_epochs": num_epochs,
            "patience": patience,
        }
        if USE_CONV_MODEL:
            parameters["conv_kwargs"] = conv_kwargs
        else:
            parameters["feed_forward_kwargs"] = feed_forward_kwargs
        parameters_str = json.dumps(parameters)

        mimo_metrics = train_stability_mimo_model(
            hidden_dim=hidden_dim,
            feed_forward_kwargs=feed_forward_kwargs,
            conv_kwargs=conv_kwargs,
            num_inputs=num_inputs,
            bs=bs,
            lr=lr,
            num_epochs=num_epochs,
            patience=patience,
            plot=False,
        )
        with open(
            f"results/stability/mimo_results/parameters={parameters_str}.json", "w"
        ) as mimo_results_path:
            json.dump(
                {"parameters": parameters, "metrics": mimo_metrics}, mimo_results_path
            )

        ensemble_metrics = train_stability_ensemble_models(
            hidden_dim=hidden_dim,
            feed_forward_kwargs=feed_forward_kwargs,
            conv_kwargs=conv_kwargs,
            num_inputs=num_inputs,
            bs=bs,
            lr=lr,
            num_epochs=num_epochs,
            patience=patience,
            plot=False,
        )
        with open(
            f"results/stability/ensemble_results/parameters={parameters_str}.json", "w"
        ) as ensemble_results_path:
            json.dump(
                {"parameters": parameters, "metrics": ensemble_metrics},
                ensemble_results_path,
            )

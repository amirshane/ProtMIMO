"""Training ProtMIMOOracle for Fluorescence data."""

import os
import argparse
import json

import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from pprint import pprint
from tape.datasets import LMDBDataset

from model import ProtMIMOOracle
from data_utils import (
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


GFP_SEQ_LEN = 237
GFP_AMINO_ACID_VOCABULARY = [
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
GFP_ALPHABET = {}
for i, aa in enumerate(GFP_AMINO_ACID_VOCABULARY):
    GFP_ALPHABET[aa] = i


def gfp_dataset_to_df(in_name):
    dataset = LMDBDataset(in_name)
    df = pd.DataFrame(list(dataset)[:])
    df["log_fluorescence"] = df.log_fluorescence.apply(lambda x: x[0])
    return df


def get_gfp_dfs():
    train_df = gfp_dataset_to_df("fluorescence/fluorescence_train.lmdb")
    val_df = gfp_dataset_to_df("fluorescence/fluorescence_valid.lmdb")
    test_df = gfp_dataset_to_df("fluorescence/fluorescence_test.lmdb")
    return train_df, val_df, test_df


try:
    train_df, val_df, test_df = get_gfp_dfs()
except FileNotFoundError:
    os.system(
        "wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz"
    )
    os.system("tar xzf fluorescence.tar.gz")
    train_df, val_df, test_df = get_gfp_dfs()


def compute_gfp_scalar_metrics(targets, preds, prefix=None, test_df=test_df):
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

    bright_test_df, dark_test_df = (
        test_df[test_df.log_fluorescence > 2.5],
        test_df[test_df.log_fluorescence <= 2.5],
    )

    bright_spearmanr = spearmanr(
        bright_test_df.targets.values, bright_test_df.preds.values
    ).correlation
    bright_mse = mean_squared_error(
        bright_test_df.targets.values, bright_test_df.preds.values
    )
    scalar_metrics[f"{prefix}bright_spearmanr"] = bright_spearmanr
    scalar_metrics[f"{prefix}bright_mse"] = bright_mse

    dark_spearmanr = spearmanr(
        dark_test_df.targets.values, dark_test_df.preds.values
    ).correlation
    dark_mse = mean_squared_error(
        dark_test_df.targets.values, dark_test_df.preds.values
    )
    scalar_metrics[f"{prefix}dark_spearmanr"] = dark_spearmanr
    scalar_metrics[f"{prefix}dark_mse"] = dark_mse

    for metric_key in scalar_metrics.keys():
        scalar_metrics[metric_key] = float(scalar_metrics[metric_key])

    return scalar_metrics


def get_gfp_metrics(
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
        compute_gfp_scalar_metrics(
            targets=targets, preds=preds, prefix=None, test_df=test_df
        )
    )
    for i in range(num_inputs):
        metrics[f"model_{i}_test_loss"] = loss_fn(
            torch.tensor(targets), torch.tensor(preds_by_input[f"model_{i}"])
        ).item()
        metrics.update(
            compute_gfp_scalar_metrics(
                targets=targets, preds=preds, prefix=f"model_{i}", test_df=test_df
            )
        )
        for j in range(i + 1, num_inputs):
            metrics[f"model_{i}_{j}_residual_correlation"] = pearsonr(
                preds_by_input[f"model_{i}"], preds_by_input[f"model_{j}"]
            )[0]

    if plot:
        if ensemble:
            title = f"Ensemble of {num_inputs} CNN Models"
            path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/ensemble_of_{num_inputs}_models.jpg'
        else:
            title = f"MIMO CNN Model with {num_inputs} Inputs and Outputs"
            path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/mimo_with_{num_inputs}_inputs.jpg'
        create_plot(
            targets=targets,
            preds=preds,
            title=title,
            path=path,
            feature_name="Log-Fluorescence",
        )

        for i in range(num_inputs):
            if ensemble:
                title = f"CNN Model {i + 1} of Ensemble of {num_inputs} Models"
                path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/model_{i+1}_of_ensemble_of_{num_inputs}_models.jpg'
            else:
                title = f"Output {i + 1} of MIMO CNN Model with {num_inputs} Inputs and Outputs"
                path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/output_{i+1}_of_mimo_with_{num_inputs}_inputs.jpg'
            create_plot(
                targets,
                preds_by_input[f"model_{i}"],
                title=title,
                path=path,
                feature_name="Log-Fluorescence",
            )

    return metrics


# MIMO Training/Evaluation
def train_gfp_mimo_model(
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
        train_df=train_df, num_inputs=num_inputs, bs=bs, feature_name="log_fluorescence"
    )
    val_data = create_batched_train_data(
        train_df=val_df, num_inputs=num_inputs, bs=bs, feature_name="log_fluorescence"
    )
    test_data = create_batched_test_data(
        test_df=test_df, num_inputs=num_inputs, bs=bs, feature_name="log_fluorescence"
    )
    data = {
        "training_data": training_data,
        "val_data": val_data,
        "test_data": test_data,
    }

    if USE_CONV_MODEL:
        model = ProtMIMOOracle(
            alphabet=GFP_ALPHABET,
            max_len=GFP_SEQ_LEN,
            num_inputs=num_inputs,
            hidden_dim=hidden_dim,
            convolutional=True,
            conv_kwargs=conv_kwargs,
        )
    else:
        model = ProtMIMOOracle(
            alphabet=GFP_ALPHABET,
            max_len=GFP_SEQ_LEN,
            num_inputs=num_inputs,
            hidden_dim=hidden_dim,
            feed_forward_kwargs=feed_forward_kwargs,
        )
    model.to(DEVICE)

    best_model, test_metrics = train_and_evaluate(
        model=model,
        data=data,
        metrics_fn=get_gfp_metrics,
        plot=plot,
        num_inputs=num_inputs,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
    )
    return test_metrics


# Standard Ensemble
def train_gfp_ensemble_models(
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
            feature_name="log_fluorescence",
            ensemble_model_num=i,
        )
        ensemble_val_data = create_batched_train_data(
            train_df=val_df,
            num_inputs=1,
            bs=bs,
            feature_name="log_fluorescence",
            ensemble_model_num=i,
        )
        ensemble_test_data = create_batched_test_data(
            test_df=test_df, num_inputs=1, bs=bs, feature_name="log_fluorescence"
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
                alphabet=GFP_ALPHABET,
                max_len=GFP_SEQ_LEN,
                num_inputs=1,
                hidden_dim=hidden_dim,
                convolutional=True,
                conv_kwargs=conv_kwargs,
            )
            if USE_CONV_MODEL
            else ProtMIMOOracle(
                alphabet=GFP_ALPHABET,
                max_len=GFP_SEQ_LEN,
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
        metrics_fn=get_gfp_metrics,
        plot=plot,
        lr=lr,
        num_epochs=num_epochs,
        patience=patience,
    )
    return test_metrics


if __name__ == "__main__":
    hidden_dim = 512
    bs = 32
    lr = 0.0001
    num_epochs = 100
    patience = 10
    job = 0
    for num_inputs in [2, 3, 4, 5]:
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

            mimo_metrics = train_gfp_mimo_model(
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
#            with open(
#                f"results/fluorescence/mimo_results/parameters={parameters_str}.json",
#                "w",
#            ) as mimo_results_path:
#                json.dump(
#                    {"parameters": parameters, "metrics": mimo_metrics},
#                    mimo_results_path,
#                )
            with open(
                f"results//fluorescence//mimo_results//job_{job}.json",
                "w",
            ) as mimo_results_path:
                json.dump(
                    {"parameters": parameters, "metrics": mimo_metrics},
                    mimo_results_path,
                )

            ensemble_metrics = train_gfp_ensemble_models(
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
#            with open(
#                f"results/fluorescence/ensemble_results/parameters={parameters_str}.json",
#                "w",
#            ) as ensemble_results_path:
#                json.dump(
#                    {"parameters": parameters, "metrics": ensemble_metrics},
#                    ensemble_results_path,
#                )
            with open(
                f"results//fluorescence//ensemble_results//job_{job}.json",
                "w",
            ) as ensemble_results_path:
                json.dump(
                    {"parameters": parameters, "metrics": ensemble_metrics},
                    ensemble_results_path,
                )
            
            job += 1

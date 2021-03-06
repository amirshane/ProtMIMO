"""Shared data utils."""

import copy
from math import ceil
import enum

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetType(enum.Enum):
    Train = "train"
    Val = "val"
    Test = "test"


def create_batched_train_data(
    train_df, num_inputs, bs, feature_name, ensemble_model_num=None
):
    N = len(train_df)
    random_states = (
        [42 + i for i in range(num_inputs)]
        if num_inputs > 1
        else [42 + ensemble_model_num]
    )

    shuffled_dfs = [
        train_df.sample(frac=1, random_state=random_states[i])
        for i in range(num_inputs)
    ]
    data = []
    for i in range(ceil(N / bs)):
        batch_data = []
        for j in range(num_inputs):
            shuffled_batch_df = shuffled_dfs[j].iloc[i * bs : min((i + 1) * bs, N)]
            x_j = shuffled_batch_df.primary.values
            y_j = shuffled_batch_df[feature_name].values
            batch_data.append((x_j, y_j))
        x = [
            [batch_data[j][0][k] for j in range(num_inputs)]
            for k in range(len(batch_data[0][0]))
        ]
        y = [
            [batch_data[j][1][k] for j in range(num_inputs)]
            for k in range(len(batch_data[0][0]))
        ]
        data.append((np.array(x), np.array(y)))
    return data


def create_batched_test_data(test_df, num_inputs, bs, feature_name):
    N = len(test_df)
    data = []
    for i in range(ceil(N / bs)):
        batch_df = test_df.iloc[i * bs : min((i + 1) * bs, N)]
        batch_data = [
            (batch_df.primary.values, batch_df[feature_name].values)
        ] * num_inputs
        x = [
            [batch_data[j][0][k] for j in range(num_inputs)]
            for k in range(len(batch_data[0][0]))
        ]
        y = [
            [batch_data[j][1][k] for j in range(num_inputs)]
            for k in range(len(batch_data[0][0]))
        ]
        data.append((np.array(x), np.array(y)))
    return data


def create_plot(targets, preds, title, path, feature_name, show_plot=False):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, preds, alpha=0.15)
    plt.xlabel(f"True {feature_name}")
    plt.ylabel(f"Predicted {feature_name}")
    plt.title(title)
    plt.savefig(path)
    if show_plot:
        plt.show()


def validate(model, loss_fn, data, ensemble=False):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(data):
            inputs, targets = batch
            if ensemble and targets.shape[0] == 1:
                targets = targets[0]
            else:
                targets = nn.Flatten(0)(torch.tensor(targets)).squeeze().numpy()
            all_targets += list(targets)

            preds = model(inputs)
            if ensemble and preds.shape[0] == 1:
                preds = preds[0][0]
            else:
                preds = nn.Flatten(0)(preds).squeeze().cpu().numpy()
            all_preds += list(preds)
    loss = loss_fn(torch.tensor(all_targets), torch.tensor(all_preds)).item()
    return loss


def mimo_evaluate(model, num_inputs, loss_fn, test_data, metrics_fn, plot=False):
    preds_by_input = {}
    for i in range(num_inputs):
        preds_by_input[f"model_{i}"] = []

    model.eval()
    test_targets, test_preds = [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(test_data):
            inputs, targets = batch
            targets = targets[:, 0]
            test_targets += list(targets)

            preds = model(inputs)
            for i in range(num_inputs):
                preds_by_input[f"model_{i}"] += list(
                    preds.squeeze().cpu().numpy()[:, i]
                )
            preds = torch.mean(preds, 1).squeeze().cpu().numpy()
            test_preds += list(preds)

    metrics = metrics_fn(
        targets=test_targets,
        preds=test_preds,
        preds_by_input=preds_by_input,
        num_inputs=num_inputs,
        plot=plot,
        loss_fn=loss_fn,
    )
    return metrics


def ensemble_evaluate(models, loss_fn, test_data, metrics_fn, plot=False):
    num_inputs = len(models)

    preds_by_input = {}
    for i in range(num_inputs):
        preds_by_input[f"model_{i}"] = []

    test_targets = []
    for i in range(num_inputs):
        model = models[i]
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(test_data):
                inputs, targets = batch
                if i == 0:
                    targets = targets[:, 0]
                    test_targets += list(targets)

                preds = model(inputs)
                preds_by_input[f"model_{i}"] += list(preds.squeeze().cpu().numpy())
    test_preds = np.mean(
        np.array([preds_by_input[f"model_{i}"] for i in range(num_inputs)]), axis=0
    )

    metrics = metrics_fn(
        targets=test_targets,
        preds=test_preds,
        preds_by_input=preds_by_input,
        num_inputs=num_inputs,
        loss_fn=loss_fn,
        plot=plot,
        ensemble=True,
    )
    return metrics


def train_and_evaluate(
    model,
    data,
    metrics_fn,
    plot=False,
    num_inputs=1,
    lr=0.001,
    num_epochs=100,
    patience=10,
    evaluate=True,
    ensemble=False,
    ensemble_model_num=None,
):
    best_model = copy.deepcopy(model)

    training_data = data["training_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    patience_count, min_val_loss = 0, 0.0
    ensemble_model_num_str = (
        f" for Model {ensemble_model_num}" if ensemble_model_num is not None else ""
    )

    training_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        for batch_num, batch in enumerate(training_data):
            inputs, targets = batch
            targets = nn.Flatten(0)(torch.tensor(targets))

            preds = model(inputs)
            preds = nn.Flatten(0)(preds)

            loss = loss_fn(preds.to(DEVICE), targets.to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = validate(
            model=model, loss_fn=loss_fn, data=training_data, ensemble=ensemble
        )
        training_losses.append(train_loss)
        print(
            f"Train Loss{ensemble_model_num_str} at Epoch {epoch}: {round(train_loss, 3)}"
        )
        val_loss = validate(
            model=model, loss_fn=loss_fn, data=val_data, ensemble=ensemble
        )
        val_losses.append(val_loss)
        print(
            f"Validation Loss{ensemble_model_num_str} at Epoch {epoch}: {round(val_loss, 3)}"
        )
        if epoch == 0:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        else:
            if val_loss >= min_val_loss:
                patience_count += 1
            else:
                min_val_loss = val_loss
                patience_count = 0
                best_model = copy.deepcopy(model)
        print(f"Patience Count{ensemble_model_num_str}: {patience_count}")
        if patience_count >= patience:
            break
        print()

    if evaluate:
        test_metrics = mimo_evaluate(
            model=best_model,
            num_inputs=num_inputs,
            loss_fn=loss_fn,
            test_data=test_data,
            metrics_fn=metrics_fn,
            plot=plot,
        )
        if ensemble:
            return best_model, test_metrics, training_losses, val_losses
        else:
            test_metrics["training_losses"] = training_losses
            test_metrics["val_losses"] = val_losses
            return best_model, test_metrics
    else:
        return best_model, training_losses, val_losses


def train_and_evaluate_ensemble(
    models,
    data,
    metrics_fn,
    plot=False,
    lr=0.001,
    num_epochs=100,
    patience=10,
    evaluate=True,
):
    best_models = copy.deepcopy(models)
    ensemble_training_losses, ensemble_val_losses = {}, {}

    for i, model in enumerate(models):
        best_model, training_losses, val_losses = train_and_evaluate(
            model=model,
            data=data[i],
            metrics_fn=None,
            plot=plot,
            lr=lr,
            num_epochs=num_epochs,
            patience=patience,
            evaluate=False,
            ensemble=True,
            ensemble_model_num=i,
        )
        best_models[i] = best_model
        ensemble_training_losses[f"ensemble_model_{i}"] = training_losses
        ensemble_val_losses[f"ensemble_model_{i}"] = val_losses

    if evaluate:
        test_metrics = ensemble_evaluate(
            models=best_models,
            loss_fn=nn.MSELoss(),
            test_data=data[0]["test_data"],
            metrics_fn=metrics_fn,
            plot=plot,
        )
        test_metrics["training_losses"] = ensemble_training_losses
        test_metrics["val_losses"] = ensemble_val_losses
        return best_models, test_metrics
    else:
        return best_models

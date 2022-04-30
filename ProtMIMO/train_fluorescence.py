"""Traing ProtMIMOOracle for Fluorescence data."""

import os
import copy
from math import ceil
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error
from pprint import pprint
import matplotlib.pyplot as plt

from model import ProtMIMOOracle
import tape
from tape.datasets import LMDBDataset


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Determine model type.')
parser.add_argument('convolutional', type=bool, nargs='?', default=False)
args = parser.parse_args()
USE_CONV_MODEL = args.convolutional


GFP_SEQ_LEN = 237
GFP_AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', '.'
]
GFP_ALPHABET = {}
for i, aa in enumerate(GFP_AMINO_ACID_VOCABULARY):
    GFP_ALPHABET[aa] = i

def gfp_dataset_to_df(in_name):
  dataset = LMDBDataset(in_name)
  df = pd.DataFrame(list(dataset)[:])
  df['log_fluorescence'] = df.log_fluorescence.apply(lambda x: x[0])
  return df

def get_gfp_dfs():
    train_df = gfp_dataset_to_df('fluorescence/fluorescence_train.lmdb')
    shuffled_train_df = train_df.sample(frac=1, random_state=11)
    N = len(train_df)
    train_df, val_df = shuffled_train_df.iloc[: 9 * N // 10], shuffled_train_df.iloc[9 * N // 10 :]
    test_df = gfp_dataset_to_df('fluorescence/fluorescence_test.lmdb')
    return train_df, val_df, test_df

try:
    train_df, val_df, test_df = get_gfp_dfs()
except FileNotFoundError:
    os.system('wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz')
    os.system('tar xzf fluorescence.tar.gz')
    train_df, val_df, test_df = get_gfp_dfs()


def create_batched_gfp_train_data(train_df=train_df, num_inputs=3, bs=32):
    N = len(train_df)
    random_states = [42 + i for i in range(num_inputs)]
    shuffled_dfs = [train_df.sample(frac=1, random_state=random_states[i]) for i in range(num_inputs)]
    data = []
    for i in range(ceil(N / bs)):
        batch_data = []
        for j in range(num_inputs):
            shuffled_batch_df = shuffled_dfs[j].iloc[i * bs : min((i+1) * bs, N)]
            x_j = shuffled_batch_df.primary.values
            y_j = shuffled_batch_df.log_fluorescence.values
            batch_data.append((x_j, y_j))
        x = [[batch_data[j][0][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        y = [[batch_data[j][1][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        data.append((np.array(x), np.array(y)))
    return data


def create_batched_gfp_test_data(test_df=test_df, num_inputs=3, bs=32):
    N = len(test_df)
    data = []
    for i in range(ceil(N / bs)):
        batch_df = test_df.iloc[i * bs : min((i+1) * bs, N)]
        batch_data = [(batch_df.primary.values, batch_df.log_fluorescence.values) for _ in range(num_inputs)]
        x = [[batch_data[j][0][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        y = [[batch_data[j][1][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        data.append((np.array(x), np.array(y)))
    return data


def create_plot(targets, preds, title, path):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, preds, alpha=0.15)
    plt.xlabel('True Log-Fluorescence')
    plt.ylabel('Predicted Log-Fluorescence')
    plt.title(title)
    plt.savefig(path)
    # plt.show()


def compute_scalar_metrics(targets, preds, prefix=None, test_df=test_df):
    scalar_metrics = {}
    if prefix:
        prefix = f'{prefix}_'
    else:
        prefix = ''
    test_df['targets'] = targets
    test_df['preds'] = preds

    base_spearmanr = spearmanr(targets, preds).correlation
    base_mse = mean_squared_error(targets, preds)
    scalar_metrics[f'{prefix}spearmanr'] = base_spearmanr
    scalar_metrics[f'{prefix}mse'] = base_mse

    bright_test_df, dark_test_df = test_df[test_df.log_fluorescence > 2.5], test_df[test_df.log_fluorescence <= 2.5]
    
    bright_spearmanr = spearmanr(bright_test_df.targets.values, bright_test_df.preds.values).correlation
    bright_mse = mean_squared_error(bright_test_df.targets.values, bright_test_df.preds.values)
    scalar_metrics[f'{prefix}bright_spearmanr'] = bright_spearmanr
    scalar_metrics[f'{prefix}bright_mse'] = bright_mse
    
    dark_spearmanr = spearmanr(dark_test_df.targets.values,
    dark_test_df.preds.values).correlation
    dark_mse = mean_squared_error(dark_test_df.targets.values, dark_test_df.preds.values)
    scalar_metrics[f'{prefix}dark_spearmanr'] = dark_spearmanr
    scalar_metrics[f'{prefix}dark_mse'] = dark_mse
    
    return scalar_metrics
    


def get_metrics(targets, preds, preds_by_input, num_inputs, loss_fn, test_df=test_df, ensemble=False):
    metrics = {}
    test_loss = loss_fn(torch.tensor(targets), torch.tensor(preds)).item()
    metrics['test_loss'] = test_loss
    metrics.update(compute_scalar_metrics(targets=targets, preds=preds, prefix=None, test_df=test_df))
    for i in range(num_inputs):
        metrics[f'model_{i}_test_loss'] = loss_fn(torch.tensor(targets), torch.tensor(preds_by_input[f'model_{i}'])).item()
        metrics.update(compute_scalar_metrics(targets=targets, preds=preds, prefix=f'model_{i}', test_df=test_df))
        for j in range(i+1, num_inputs):
            metrics[f'model_{i}_{j}_residual_correlation'] = pearsonr(preds_by_input[f'model_{i}'], preds_by_input[f'model_{j}'])[0]
    
    if ensemble:
        title = f'Ensemble of {num_inputs} CNN Models'
        path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/ensemble_of_{num_inputs}_models.jpg'
    else:
        title = f'MIMO CNN Model with {num_inputs} Inputs and Outputs'
        path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/mimo_with_{num_inputs}_inputs.jpg'
    create_plot(targets=targets, preds=preds, title=title, path=path)
    
    for i in range(num_inputs):
        if ensemble:
            title = f'CNN Model {i + 1} of Ensemble of {num_inputs} Models'
            path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/model_{i+1}_of_ensemble_of_{num_inputs}_models.jpg'
        else:
            title = f'Output {i + 1} of MIMO CNN Model with {num_inputs} Inputs and Outputs'
            path = f'fluorescence_figures/{"convolutional" if USE_CONV_MODEL else "feed_forward"}/output_{i+1}_of_mimo_with_{num_inputs}_inputs.jpg'
        create_plot(targets, preds_by_input[f'model_{i}'], title=title, path=path)
    
    return metrics


def validate(model, loss_fn, val_data, ensemble=False):
    model.eval()
    val_targets, val_preds = [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(val_data):
            inputs, targets = batch
            if ensemble and targets.shape[0] == 1:
                targets = targets[0]
            else:
                targets = nn.Flatten(0)(torch.tensor(targets)).squeeze().numpy()
            val_targets += list(targets)

            preds = model(inputs)
            if ensemble and preds.shape[0] == 1:
                preds = preds[0][0]
            else:
                preds = nn.Flatten(0)(preds).squeeze().cpu().numpy()
            val_preds += list(preds)
    val_loss = loss_fn(torch.tensor(val_targets), torch.tensor(val_preds)).item()
    return val_loss


def evaluate(model, num_inputs, loss_fn, test_data):
    preds_by_input = {}
    for i in range(num_inputs):
        preds_by_input[f'model_{i}'] = []

    model.eval()
    test_targets, test_preds = [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(test_data):
            inputs, targets = batch
            targets = targets[:, 0]
            test_targets += list(targets)

            preds = model(inputs)
            for i in range(num_inputs):
                preds_by_input[f'model_{i}'] += list(preds.squeeze().cpu().numpy()[:, i])
            preds = torch.mean(preds, 1).squeeze().cpu().numpy()
            test_preds += list(preds)
    
    metrics = get_metrics(targets=test_targets, preds=test_preds, preds_by_input=preds_by_input,
                          num_inputs=num_inputs, loss_fn=loss_fn)
    return metrics


def ensemble_evaluate(models, loss_fn, test_data):
    num_inputs = len(models)

    preds_by_input = {}
    for i in range(num_inputs):
        preds_by_input[f'model_{i}'] = []

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
                preds_by_input[f'model_{i}'] += list(preds.squeeze().cpu().numpy())
    test_preds = np.mean(np.array([preds_by_input[f'model_{i}'] for i in range(num_inputs)]), axis=0)

    metrics = get_metrics(targets=test_targets, preds=test_preds, preds_by_input=preds_by_input,
                          num_inputs=num_inputs, loss_fn=loss_fn, ensemble=True)
    return metrics


# MIMO Training/Evaluation
num_inputs = 3
bs = 32
training_data = create_batched_gfp_train_data(train_df=train_df, num_inputs=num_inputs, bs=bs)
val_data = create_batched_gfp_train_data(train_df=val_df, num_inputs=num_inputs, bs=bs)
test_data = create_batched_gfp_test_data(test_df=test_df, num_inputs=num_inputs, bs=bs)


if USE_CONV_MODEL:
    model = ProtMIMOOracle(
        alphabet=GFP_ALPHABET,
        max_len=GFP_SEQ_LEN,
        num_inputs=num_inputs,
        hidden_dim=512,
        convolutional=True,
        conv_kwargs=
            {
                'channels': [32, 16, 8],
                'kernel_sizes': [7, 5, 3],
                'pooling_dims': [3, 2, 0],
            }
    )
else:
    model = ProtMIMOOracle(
        alphabet=GFP_ALPHABET,
        max_len=GFP_SEQ_LEN,
        num_inputs=num_inputs,
        hidden_dim=512,
        feed_forward_kwargs=
            {
                'hidden_dims': [256, 128, 64],
            }
    )
model.to(DEVICE)
best_model = copy.deepcopy(model)

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
num_epochs = 100
patience, patience_count, min_val_loss = 10, 0, 0.0
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

    val_loss = validate(model=model, loss_fn=loss_fn, val_data=val_data)
    print(f'Validation Loss at Epoch {epoch}: {round(val_loss, 3)}')
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
    print(f'Patience Count: {patience_count}')
    if patience_count >= patience:
        break
    print()

test_metrics = evaluate(model=best_model, num_inputs=num_inputs, loss_fn=loss_fn, test_data=test_data)
pprint(test_metrics)
print()


# Standard Ensemble
ensemble_training_data = create_batched_gfp_train_data(train_df=train_df, num_inputs=1, bs=bs)
ensemble_val_data = create_batched_gfp_train_data(train_df=val_df, num_inputs=1, bs=bs)
ensemble_test_data = create_batched_gfp_test_data(test_df=test_df, num_inputs=1, bs=bs)

models = [
    (ProtMIMOOracle(
        alphabet=GFP_ALPHABET,
        max_len=GFP_SEQ_LEN,
        num_inputs=1,
        hidden_dim=512,
        convolutional=True,
        conv_kwargs=
            {
                'channels': [32, 16, 8],
                'kernel_sizes': [7, 5, 3],
                'pooling_dims': [3, 2, 0],
            }
    ) if USE_CONV_MODEL else
    ProtMIMOOracle(
        alphabet=GFP_ALPHABET,
        max_len=GFP_SEQ_LEN,
        num_inputs=1,
        hidden_dim=512,
        feed_forward_kwargs=
            {
                'hidden_dims': [256, 128, 64],
            }
    )).to(DEVICE)
    for _ in range(num_inputs)
]

best_models = copy.deepcopy(models)

num_epochs = 100
patience, patience_counts, min_val_losses = 10, [0, 0, 0], [0.0, 0.0, 0.0]

for i, model in enumerate(models):
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        for batch_num, batch in enumerate(ensemble_training_data):
            inputs, targets = batch
            targets = nn.Flatten(0)(torch.tensor(targets))

            preds = model(inputs)
            preds = nn.Flatten(0)(preds)

            loss = loss_fn(preds.to(DEVICE), targets.to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = validate(model=model, loss_fn=loss_fn, val_data=ensemble_val_data, ensemble=True)
        print(f'Validation Loss for Model {i} at Epoch {epoch}: {round(val_loss, 3)}')
        if epoch == 0:
            min_val_losses[i] = val_loss
            best_models[i] = copy.deepcopy(model)
        else:
            if val_loss >= min_val_losses[i]:
                patience_counts[i] += 1
            else:
                min_val_losses[i] = val_loss
                patience_counts[i] = 0
                best_models[i] = copy.deepcopy(model)
        print(f'Patience Count for Model {i}: {patience_counts[i]}')
        if patience_counts[i] >= patience:
            break
    models[i] = model
    print()

test_metrics = ensemble_evaluate(models=best_models, loss_fn=loss_fn, test_data=ensemble_test_data)
pprint(test_metrics)

"""Traing ProtMIMOOracle for Fluorescence data."""

import os
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, pearsonr
from pprint import pprint

from model import ProtMIMOOracle
import tape
from tape.datasets import LMDBDataset


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
    for i in range(N // bs):
        batch_data = []
        for j in range(num_inputs):
            shuffled_batch_df = shuffled_dfs[j].iloc[i * bs : min((i+1) * bs, N-1)]
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
    for i in range(N // bs):
        batch_df = test_df.iloc[i * bs : min((i+1) * bs, N-1)]
        batch_data = [(batch_df.primary.values, batch_df.log_fluorescence.values) for _ in range(num_inputs)]
        x = [[batch_data[j][0][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        y = [[batch_data[j][1][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        data.append((np.array(x), np.array(y)))
    return data


def get_metrics(targets, preds, preds_by_input, num_inputs, loss_fn):
    metrics = {}
    test_loss = loss_fn(torch.tensor(targets), torch.tensor(preds)).item()
    metrics['test_loss'] = test_loss
    metrics['spearmanr'] = spearmanr(targets, preds).correlation
    for i in range(num_inputs):
        metrics[f'model_{i}_test_loss'] = loss_fn(torch.tensor(targets), torch.tensor(preds_by_input[f'model_{i}'])).item()
        metrics[f'model_{i}_spearmanr'] = spearmanr(targets, preds_by_input[f'model_{i}']).correlation
        for j in range(i+1, num_inputs):
            metrics[f'model_{i}_{j}_residual_correlation'] = pearsonr(preds_by_input[f'model_{i}'], preds_by_input[f'model_{j}'])[0]
    return metrics


def validate(model, loss_fn, val_data):
    model.eval()
    val_targets, val_preds = [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(val_data):
            inputs, targets = batch
            targets = nn.Flatten(0)(torch.tensor(targets)).squeeze().numpy()
            val_targets += list(targets)

            preds = model(inputs)
            preds = nn.Flatten(0)(preds).squeeze().numpy()
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
                preds_by_input[f'model_{i}'] += list(preds.squeeze().numpy()[:, i])
            preds = torch.mean(preds, 1).squeeze().numpy()
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
                preds_by_input[f'model_{i}'] += list(preds.squeeze().numpy())
    test_preds = np.mean(np.array([preds_by_input[f'model_{i}'] for i in range(num_inputs)]), axis=0)

    metrics = get_metrics(targets=test_targets, preds=test_preds, preds_by_input=preds_by_input,
                          num_inputs=num_inputs, loss_fn=loss_fn)
    return metrics


# MIMO Training/Evaluation
num_inputs = 3
bs = 32
training_data = create_batched_gfp_train_data(train_df=train_df, num_inputs=num_inputs, bs=bs)
val_data = create_batched_gfp_train_data(train_df=val_df, num_inputs=num_inputs, bs=bs)
test_data = create_batched_gfp_test_data(test_df=test_df, num_inputs=num_inputs, bs=bs)

model = ProtMIMOOracle(
    alphabet=GFP_ALPHABET,
    max_len=GFP_SEQ_LEN,
    num_inputs=num_inputs,
    hidden_dim=512,
    channels=[32, 16, 8],
    kernel_sizes=[7, 5, 3],
    pooling_dims=[3, 2, 0],
)
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

        loss = loss_fn(preds, targets)

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

test_metrics = evaluate(model=model, num_inputs=num_inputs, loss_fn=loss_fn, test_data=test_data)
pprint(test_metrics)
print()


# Standard Ensemble
ensemble_training_data = create_batched_gfp_train_data(train_df=train_df, num_inputs=1, bs=bs)
ensemble_val_data = create_batched_gfp_train_data(train_df=val_df, num_inputs=1, bs=bs)
ensemble_test_data = create_batched_gfp_test_data(test_df=test_df, num_inputs=1, bs=bs)

models = [
    ProtMIMOOracle(
        alphabet=GFP_ALPHABET,
        max_len=GFP_SEQ_LEN,
        num_inputs=1,
        hidden_dim=512,
        channels=[32, 16, 8],
        kernel_sizes=[7, 5, 3],
        pooling_dims=[3, 2, 0],
    ) for _ in range(num_inputs)
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

            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = validate(model=model, loss_fn=loss_fn, val_data=ensemble_val_data)
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

test_metrics = ensemble_evaluate(models=models, loss_fn=loss_fn, test_data=ensemble_test_data)
pprint(test_metrics)

"""Traing ProtMIMOOracle for Stability data."""

import os
import copy
from math import ceil

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


STABILITY_SEQ_LEN = 50
STABILITY_AMINO_ACID_VOCABULARY = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y', '.'
]
STABILITY_ALPHABET = {}
for i, aa in enumerate(STABILITY_AMINO_ACID_VOCABULARY):
    STABILITY_ALPHABET[aa] = i


def stability_dataset_to_df(in_name):
  dataset = LMDBDataset(in_name)
  df = pd.DataFrame(list(dataset)[:])
  df['stability'] = df.stability_score.apply(lambda x: x[0])
  df['id_str'] = df.id.apply(lambda x: x.decode('utf-8'))
  return df

try:
    stability_train_df = stability_dataset_to_df('stability/stability_train.lmdb')
    stability_test_df = stability_dataset_to_df('stability/stability_test.lmdb')
except FileNotFoundError:
    os.system('wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/stability.tar.gz')
    os.system('tar xzf stability.tar.gz')
    stability_train_df = stability_dataset_to_df('stability/stability_train.lmdb')
    stability_test_df = stability_dataset_to_df('stability/stability_test.lmdb')

parent_to_parent_stability = {}
for parent in set(stability_train_df.parent.values):
  stabilities = stability_train_df[stability_train_df['id_str']==parent.decode('utf-8') + '.pdb'].stability.values
  if len(stabilities) == 0:
    stabilities = stability_train_df[stability_train_df['id_str']==parent.decode('utf-8')].stability.values
    if len(stabilities) == 0:
      parent_to_parent_stability[parent] = None
    else:
      parent_to_parent_stability[parent] = stabilities[0]
  else:
    parent_to_parent_stability[parent] = stabilities[0]
for parent in set(stability_test_df.parent.values):
  stabilities = stability_test_df[stability_test_df['id_str']==parent.decode('utf-8') + '.pdb'].stability.values
  if len(stabilities) == 0:
    stabilities = stability_test_df[stability_test_df['id_str']==parent.decode('utf-8')].stability.values
    if len(stabilities) == 0:
      parent_to_parent_stability[parent] = None
    else:
      parent_to_parent_stability[parent] = stabilities[0]
  else:
    parent_to_parent_stability[parent] = stabilities[0]

topology_to_ind = {'HHH': 0, 'HEEH': 1, 'EHEE': 3, 'EEHEE': 4}
def topology_to_index(top):
  top = top.decode('utf-8')
  if top in topology_to_ind.keys():
    return topology_to_ind[top]
  else:
    return 2


def create_stability_df(test=False):
  if test:
    stability_df = stability_dataset_to_df('stability/stability_test.lmdb')
  else:
    stability_df = stability_dataset_to_df('stability/stability_train.lmdb')

  stability_df['parent_stability'] = stability_df.parent.apply(lambda x: parent_to_parent_stability[x])
  stability_df['topology_ind'] = stability_df.topology.apply(lambda x: topology_to_index(x))

  return stability_df


def get_stability_dfs():
    train_df = create_stability_df(test=False)
    shuffled_train_df = train_df.sample(frac=1, random_state=11)
    N = len(train_df)
    train_df, val_df = shuffled_train_df.iloc[: 9 * N // 10], shuffled_train_df.iloc[9 * N // 10 :]
    test_df = create_stability_df(test=True)
    return train_df, val_df, test_df

train_df, val_df, test_df = get_stability_dfs()


def create_batched_stability_train_data(train_df=train_df, num_inputs=3, bs=32):
    N = len(train_df)
    random_states = [42 + i for i in range(num_inputs)]
    shuffled_dfs = [train_df.sample(frac=1, random_state=random_states[i]) for i in range(num_inputs)]
    data = []
    for i in range(ceil(N / bs)):
        batch_data = []
        for j in range(num_inputs):
            shuffled_batch_df = shuffled_dfs[j].iloc[i * bs : min((i+1) * bs, N)]
            x_j = shuffled_batch_df.primary.values
            y_j = shuffled_batch_df.stability.values
            batch_data.append((x_j, y_j))
        x = [[batch_data[j][0][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        y = [[batch_data[j][1][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        data.append((np.array(x), np.array(y)))
    return data


def create_batched_stability_test_data(test_df=test_df, num_inputs=3, bs=32):
    N = len(test_df)
    data = []
    for i in range(ceil(N / bs)):
        batch_df = test_df.iloc[i * bs : min((i+1) * bs, N)]
        batch_data = [(batch_df.primary.values, batch_df.stability.values) for _ in range(num_inputs)]
        x = [[batch_data[j][0][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        y = [[batch_data[j][1][k] for j in range(num_inputs)] for k in range(len(batch_data[0][0]))]
        data.append((np.array(x), np.array(y)))
    return data


def create_plot(targets, preds, title, path):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, preds, alpha=0.15)
    plt.xlabel('True Stability')
    plt.ylabel('Predicted Stability')
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
    
    parent_to_pred_parent_stability = {}
    for parent in set(test_df.parent.values):
        pred_parent_stabilities = test_df[test_df['id_str']==parent.decode('utf-8') + '.pdb'].preds.values
        if len(pred_parent_stabilities) == 0:
            pred_parent_stabilities = test_df[test_df['id_str']==parent.decode('utf-8')].preds.values
            if len(pred_parent_stabilities) == 0:
                parent_to_pred_parent_stability[parent] = None
            else:
                parent_to_pred_parent_stability[parent] = pred_parent_stabilities[0]
        else:
            parent_to_pred_parent_stability[parent] = pred_parent_stabilities[0]
    test_df['pred_parent_stability'] = test_df.parent.apply(lambda x: parent_to_pred_parent_stability[x])

    correct_direction = 0
    for test_stability, pred_stability, parent_stability, pred_parent_stability in zip(targets, preds, test_df.parent_stability.values, test_df.pred_parent_stability.values):
        if parent_stability is not None:
            if (test_stability >= parent_stability and pred_stability >= pred_parent_stability) or (test_stability <= parent_stability and pred_stability <= pred_parent_stability):
                correct_direction += 1
    base_accuracy = correct_direction / len(np.where(test_df.parent_stability.values!=None)[0])
    scalar_metrics[f'{prefix}accuracy'] = base_accuracy
    
    test_df['topology_str'] = test_df.topology.apply(lambda x: x.decode('utf-8'))
    topologies = [('EEHEE', 'BBABB'), ('EHEE', 'BABB'), ('HEEH', 'ABBA'), ('HHH', 'AAA')]
    for topology_pair in topologies:
        topology, topology_name = topology_pair
        topology_test_df = test_df[test_df['topology_str']==topology]
        
        topology_spearmanr = spearmanr(topology_test_df.targets.values,
        topology_test_df.preds.values).correlation
        topology_mse = mean_squared_error(topology_test_df.targets.values, topology_test_df.preds.values)
        scalar_metrics[f'{prefix}{topology_name}_spearmanr'] = topology_spearmanr
        scalar_metrics[f'{prefix}{topology_name}_mse'] = topology_mse
        
        topology_correct_direction = 0
        for test_stability, pred_stability, parent_stability, pred_parent_stability in zip(topology_test_df.targets.values, topology_test_df.stability.values, topology_test_df.parent_stability.values, topology_test_df.pred_parent_stability.values):
            if parent_stability is not None:
                if (test_stability >= parent_stability and pred_stability >= pred_parent_stability) or (test_stability <= parent_stability and pred_stability <= pred_parent_stability):
                    topology_correct_direction += 1
        topology_accuracy = topology_correct_direction / len(np.where(topology_test_df.parent_stability.values!=None)[0])
        scalar_metrics[f'{prefix}{topology_name}_accuracy'] = topology_accuracy
    
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
        path = f'stability_figures/ensemble_of_{num_inputs}_models.jpg'
    else:
        title = f'MIMO CNN Model with {num_inputs} Inputs and Outputs'
        path = f'stability_figures/mimo_with_{num_inputs}_inputs.jpg'
    create_plot(targets=targets, preds=preds, title=title, path=path)
    
    for i in range(num_inputs):
        if ensemble:
            title = f'CNN Model {i + 1} of Ensemble of {num_inputs} Models'
            path = f'stability_figures/model_{i+1}_of_ensemble_of_{num_inputs}_models.jpg'
        else:
            title = f'Output {i + 1} of MIMO CNN Model with {num_inputs} Inputs and Outputs'
            path = f'stability_figures/output_{i+1}_of_mimo_with_{num_inputs}_inputs.jpg'
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
                          num_inputs=num_inputs, loss_fn=loss_fn, ensemble=True)
    return metrics


# MIMO Training/Evaluation
num_inputs = 3
bs = 32
training_data = create_batched_stability_train_data(train_df=train_df, num_inputs=num_inputs, bs=bs)
val_data = create_batched_stability_train_data(train_df=val_df, num_inputs=num_inputs, bs=bs)
test_data = create_batched_stability_test_data(test_df=test_df, num_inputs=num_inputs, bs=bs)

model = ProtMIMOOracle(
    alphabet=STABILITY_ALPHABET,
    max_len=STABILITY_SEQ_LEN,
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

test_metrics = evaluate(model=best_model, num_inputs=num_inputs, loss_fn=loss_fn, test_data=test_data)
pprint(test_metrics)
print()


# Standard Ensemble
ensemble_training_data = create_batched_stability_train_data(train_df=train_df, num_inputs=1, bs=bs)
ensemble_val_data = create_batched_stability_train_data(train_df=val_df, num_inputs=1, bs=bs)
ensemble_test_data = create_batched_stability_test_data(test_df=test_df, num_inputs=1, bs=bs)

models = [
    ProtMIMOOracle(
        alphabet=STABILITY_ALPHABET,
        max_len=STABILITY_SEQ_LEN,
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

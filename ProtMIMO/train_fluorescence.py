"""Traing ProtMIMOOracle for Fluorescence data."""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
    test_df = gfp_dataset_to_df('fluorescence/fluorescence_test.lmdb')
    return train_df, test_df

try:
    train_df, test_df = get_gfp_dfs()
except FileNotFoundError:
    os.system('wget http://s3.amazonaws.com/songlabdata/proteindata/data_pytorch/fluorescence.tar.gz')
    os.system('tar xzf fluorescence.tar.gz')
    train_df, test_df = get_gfp_dfs()


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

num_inputs = 3
bs = 32
training_data = create_batched_gfp_train_data(train_df=train_df, num_inputs=num_inputs, bs=bs)
test_data = create_batched_gfp_test_data(test_df=test_df, num_inputs=num_inputs, bs=bs)

model = ProtMIMOOracle(
    alphabet=GFP_ALPHABET,
    max_len=GFP_SEQ_LEN,
    num_inputs=num_inputs,
    channels=[32, 16, 8],
    kernel_sizes=[7, 3, 5],
    pooling_dims=[3, 2, 0],
)

lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.MSELoss()
num_epochs = 10
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

    model.eval()
    test_targets, test_preds = [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(test_data):
            inputs, targets = batch
            targets = targets[:, 0]
            test_targets += list(targets)

            preds = model(inputs)
            preds = torch.mean(preds, 1).squeeze().numpy()
            test_preds += list(preds)
    test_loss = loss_fn(torch.tensor(test_preds), torch.tensor(test_targets))
    print(test_loss)

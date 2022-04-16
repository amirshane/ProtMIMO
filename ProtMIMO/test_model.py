"""Test ProtMIMO models."""

import numpy as np
import torch
import torch.nn as nn

from model import ProtMIMOOracle


def test_oracle():
    x = [
        ['ABSJVKSUGIOPSWGUVJSLDKFGJSAKLFJAH', 'DEFERIQUWPEIROFSADFKJSHGVYSAD'],
        ['ABSJVKSUGIOPSWGUVJSLDKFGJSAKLFJAH', 'ABSJVKSUGIOPSWGUVJSLDKFGJSAKLFJAH'],
        ['KLIJDJSAFSPFUSIREOUS', 'DFJSKLAJSFPIOESUIOPFJ'],
    ]
    max_len = max([len(seq) for seqs in x for seq in seqs])
    num_inputs = len(x[0])
    model = ProtMIMOOracle(
        max_len=max_len,
        num_inputs=num_inputs,
        channels=[32, 16, 8],
        kernel_sizes=[7, 3, 5],
        pooling_dims=[3, 2, 0],
    )
    preds = model(x)

    assert((np.array(preds.shape) == np.array([len(x), num_inputs, 1])).all())
    assert(preds[1][0].item() != preds[1][1].item())


def test_oracle_training():
    x = [
        ['ABSJVKSUGIOPSWGUVJSLDKFGJSAKLFJAH', 'DEFERIQUWPEIROFSADFKJSHGVYSAD'],
        ['ABSJVKSUGIOPSWGUVJSLDKFGJSAKLFJAH', 'ABSJVKSUGIOPSWGUVJSLDKFGJSAKLFJAH'],
        ['KLIJDJSAFSPFUSIREOUS', 'DFJSKLAJSFPIOESUIOPFJ'],
    ]
    y = torch.tensor(np.array([-0.5, 0.0, -0.5, 0.5, 0.25, -0.25])).float().reshape((3, 2, 1))
    max_len = max([len(seq) for seqs in x for seq in seqs])
    num_inputs = len(x[0])
    model = ProtMIMOOracle(
        max_len=max_len,
        num_inputs=num_inputs,
        channels=[32, 16, 8],
        kernel_sizes=[7, 3, 5],
        pooling_dims=[3, 2, 0],
    )

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    training_data = [(x, y)]
    num_epochs = 10000
    model.train()
    for epoch in range(num_epochs):
        for batch in training_data:
            inputs, targets = batch
            targets = nn.Flatten(0)(targets)

            preds = model(inputs)
            preds = nn.Flatten(0)(preds)

            loss = loss_fn(preds, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    targets = nn.Flatten(0)(y).numpy()
    model.eval()
    with torch.no_grad():
        preds = model(inputs)
    preds = nn.Flatten(0)(preds).numpy()
    assert((np.abs(targets - preds) < 1e-3).all())

"""Helper NN methods."""

import torch.nn as nn


class MultiHeadLinear(nn.Module):
    """
    Multiheaded output layer.
    PyTorch implementation of
    https://github
    .com/google/edward2/blob/59bf91ca22ed4656e0a8bf0b6d22404644d7b017/edward2/tensorflow/layers/dens
    e.py#L1287
    """

    def __init__(self, in_features, out_features, num_outputs, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_outputs = num_outputs
        self.bias = bias

        self.fc = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features * self.num_outputs,
            bias=self.bias,
        )

    def forward(self, x):
        dim = x.shape[0]
        outputs = self.fc(x)
        outputs = outputs.reshape((dim, self.num_outputs, self.out_features))
        return outputs

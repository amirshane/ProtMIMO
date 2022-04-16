"""Test helper NN methods."""

import torch
import numpy as np

from modules import MultiHeadLinear


def test_multi_head_linear():
    dim = 3
    in_features = 5
    out_features = 7
    num_outputs = 11
    x = torch.rand(dim, in_features)
    model = MultiHeadLinear(
        in_features=in_features,
        out_features=out_features,
        num_outputs=num_outputs
    )
    outputs = model(x)
    assert((np.array(outputs.shape) == np.array([dim, num_outputs, out_features])).all())

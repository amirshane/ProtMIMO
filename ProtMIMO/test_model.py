import numpy as np

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

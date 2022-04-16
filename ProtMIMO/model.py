"""ProtMIMO model."""

import copy
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import MultiHeadLinear


alphabet = {}
for i, aa in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    alphabet[aa] = i
alphabet['.'] = len('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

def seq_encode(seq, alphabet, padded_length=None):
    if padded_length:
        assert(padded_length >= len(seq))
        seq = seq + '.' * (padded_length - len(seq))
    encoded_seq = np.array([alphabet[aa] for aa in seq])
    return encoded_seq


def multi_seq_encode(seqs, alphabet, max_len=None):
    if not max_len:
        max_len = max([len(seq) for seq in seqs])
    assert(max_len >= max([len(seq) for seq in seqs]))
    encoded_seqs = np.concatenate([seq_encode(seq, alphabet, max_len) for seq in seqs])
    return encoded_seqs


def batch_seq_encode(batch_seqs, alphabet, max_len=None):
    if not max_len:
        max_len = max([len(seq) for seqs in batch_seqs for seq in seqs])
    assert(max_len >= max([len(seq) for seqs in batch_seqs for seq in seqs]))
    batch_encoded_seqs = np.array([multi_seq_encode(seqs, alphabet, max_len) for seqs in batch_seqs])
    return batch_encoded_seqs


class ProtMIMOOracle(nn.Module):
    """
    MIMO oracle for proteins.
    """
    
    def __init__(self, alphabet=alphabet, max_len=None, num_inputs=1,
                 channels=[32], kernel_sizes=[5], pooling_dims=[0]):
        super().__init__()
        self.alphabet = alphabet
        self.max_len = max_len
        self.num_inputs = num_inputs
        
        conv_blocks = []
        num_conv_blocks = len(channels)
        channels = [len(self.alphabet.keys())] + channels
        conv_output_size = self.max_len * self.num_inputs
        for i in range(num_conv_blocks):
            conv_block = []
            conv_block.append(
                nn.Conv1d(
                    in_channels=channels[i],
                    out_channels=channels[i+1],
                    kernel_size=kernel_sizes[i],
                )
            )
            conv_block.append(nn.ReLU())
            conv_output_size = floor((conv_output_size - kernel_sizes[i]) / 1) + 1 # division by 1 is stride size
            if pooling_dims[i] > 0:
                conv_block.append(nn.MaxPool1d(pooling_dims[i]))
                conv_output_size /= pooling_dims[i]
            conv_blocks.append(nn.Sequential(*conv_block))

        self.conv_layers = nn.Sequential(*conv_blocks)
        self.flatten = nn.Flatten()

        self.mhl = MultiHeadLinear(conv_output_size * channels[-1], 1, self.num_inputs) # num_outputs=1 for regression
        
        
    def forward(self, x):
        # Encode sequences to concatenated tokens
        x = batch_seq_encode(x, self.alphabet, self.max_len)
        x = torch.tensor(x)

        # One-hot encode
        x = F.one_hot(x.to(torch.int64), num_classes=len(self.alphabet.keys())).float()
        x = x.permute(0, 2, 1)

        # Convolutional blocks
        x = self.conv_layers(x)
        x = self.flatten(x)

        # Multi-head linear layer
        x = self.mhl(x)

        return x

"""ProtMIMO model."""

import copy
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules import MultiHeadLinear


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
    
    def __init__(self, alphabet, max_len, num_inputs=1, hidden_dim=None,
                 channels=[32], kernel_sizes=[5], pooling_dims=[0]):
        super().__init__()
        self.alphabet = alphabet
        self.max_len = max_len
        self.num_inputs = num_inputs
        
        if hidden_dim:
            encoder_layers = []
            encoder_layers.append(nn.Linear(self.max_len * self.num_inputs, hidden_dim))
            encoder_layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*encoder_layers)
        else:
            self.encoder = None
        
        conv_blocks = []
        num_conv_blocks = len(channels)
        channels = [len(self.alphabet.keys())] + channels
        conv_output_size = hidden_dim if hidden_dim else self.max_len * self.num_inputs
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

        num_outputs = 1 # 1 for regression
        self.mhl = MultiHeadLinear(conv_output_size * channels[-1], num_outputs, self.num_inputs)

        
    def forward(self, x):
        # Encode sequences to concatenated tokens
        x = batch_seq_encode(x, self.alphabet, self.max_len)
        x = torch.tensor(x)

        # One-hot encode
        x = F.one_hot(x.to(torch.int64), num_classes=len(self.alphabet.keys())).float()
        x = x.permute(0, 2, 1)
        
        # Encode
        if self.encoder:
            x = self.encoder(x)

        # Convolutional blocks
        x = self.conv_layers(x)
        x = self.flatten(x)

        # Multi-head linear layer
        x = self.mhl(x)

        return x


class ProtMIMOFFOracle(nn.Module):
    """
    MIMO oracle for proteins.
    """
    
    def __init__(self, alphabet, max_len, num_inputs=1, hidden_dim=64,
                 hidden_dims=[32]):
        super().__init__()
        self.alphabet = alphabet
        self.max_len = max_len
        self.num_inputs = num_inputs
        self.hidden_dim = hidden_dim
        self.hidden_dims = hidden_dims
        
        encoder_layers = []
        encoder_layers.append(nn.Linear(self.max_len * self.num_inputs, self.hidden_dim))
        encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        
        ff_blocks = []
        num_ff_blocks = len(self.hidden_dims)
        for i in range(num_ff_blocks):
            input_dim = self.hidden_dim if i == 0 else self.hidden_dims[i-1]
            output_dim = self.hidden_dims[i]
            ff_block = []
            ff_block.append(nn.Linear(input_dim, output_dim))
            ff_block.append(nn.ReLU())
            ff_blocks.append(nn.Sequential(*ff_block))

        self.ff_layers = nn.Sequential(*ff_blocks)
        self.flatten = nn.Flatten()

        num_outputs = 1 # 1 for regression
        self.mhl = MultiHeadLinear(self.hidden_dims[-1] * len(self.alphabet.keys()), num_outputs, self.num_inputs)

        
    def forward(self, x):
        # Encode sequences to concatenated tokens
        x = batch_seq_encode(x, self.alphabet, self.max_len)
        x = torch.tensor(x)

        # One-hot encode
        x = F.one_hot(x.to(torch.int64), num_classes=len(self.alphabet.keys())).float()
        x = x.permute(0, 2, 1)
        
        # Encode
        x = self.encoder(x)

        # Feed-forward blocks
        x = self.ff_layers(x)
        x = self.flatten(x)

        # Multi-head linear layer
        x = self.mhl(x)

        return x

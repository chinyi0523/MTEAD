#!/usr/bin/env python3

# Copyright 2022 IIS, Academia Sinica. (author: Chin-Yi Cheng)
#

import torch
import torch.nn as nn


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, sap_dim=None):
        super(SelfAttentionPooling, self).__init__()

        if sap_dim is None:
            sap_dim = input_dim

        self.mlp = nn.Conv1d(input_dim,
                             sap_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

        self.mu = nn.Conv1d(sap_dim,
                            1,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = torch.tanh(self.mlp(x))
        w = self.softmax(self.mu(h))
        e = (x * w).sum(dim=-1)
        return e, w


class AttentiveStatisticPooling(nn.Module):
    def __init__(self, input_dim, asp_dim=None):
        super(AttentiveStatisticPooling, self).__init__()

        if asp_dim is None:
            asp_dim = input_dim

        self.bn = nn.BatchNorm1d(input_dim,
                                 eps=1e-05,
                                 momentum=0.1,
                                 affine=True,
                                 track_running_stats=True)

        self.mlp = nn.Conv1d(input_dim,
                             asp_dim,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

        self.mu = nn.Conv1d(asp_dim,
                            1,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = torch.tanh(self.mlp(self.bn(x)))
        w = self.softmax(self.mu(h))

        means = torch.sum(x * w, dim=-1)
        # stds = torch.sqrt(torch.sum(x**2 * w, dim=-1) - means**2)
        vars = torch.sum(x**2 * w, dim=-1) - means**2
        stds = torch.sqrt(torch.clamp(vars, 1e-6))

        e = torch.cat([means, stds], dim=-1)
        return e, w

#!/usr/bin/env python3

# Copyright 2022 IIS, Academia Sinica. (author: Chin-Yi Cheng)
#

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class LSTMBlock(nn.Module):
    def __init__(self,
                 num_hidden,
                 num_condition=0,
                 num_condition_group=1,
                 dropout=0.1,
                 bidirectional=False,
                 use_affine=True,
                 use_adain=False,
                 ):
        super(LSTMBlock, self).__init__()
        # Set regular layers
        self.lstm = nn.LSTM(
            num_hidden, num_hidden//2 if bidirectional else num_hidden,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.proj = nn.Linear(
            num_hidden,
            num_hidden,
        )
        self.norm = nn.Sequential(
            nn.LayerNorm(num_hidden),
            nn.Dropout(p=dropout),
        )

        # Set conditional layers
        assert num_hidden % num_condition_group == 0
        use_affine = True if use_adain else use_affine
        num_nl_hidden = num_hidden if use_adain else 1

        if num_condition > 0:
            self.cond_b = nn.Linear(
                num_condition, num_hidden//num_condition_group)
            self.cond_nl = nn.PReLU(num_parameters=num_nl_hidden, init=0.2)
        else:
            self.cond_b = None
            self.cond_nl = nn.Identity()

        if num_condition > 0 and use_affine:
            self.cond_a = nn.Linear(
                num_condition, num_hidden//num_condition_group)
        else:
            self.cond_a = None

        if num_condition > 0 and use_adain:
            self.cond_in = nn.InstanceNorm1d(num_hidden)
        else:
            self.cond_in = None

    def forward(self, X, C=None, L=None):
        ''' Bi-LSTM block for dual-path model
        Input: 
            X: Size of (Batch, Path1, Path2, Group*Dim)
            C: Size(Batch, Group, C_Dim) or Size(Batch, C_Dim)
            L: Size of (Batch, Path1)
        Output: 
            X: Size of (Batch, Path2, Path1, Group*Dim)
        '''
        Batch, Path1, Path2, Dim = X.shape
        self.lstm.flatten_parameters()
        X = X.transpose(1, 2).contiguous()
        if L is not None:
            L = L.unsqueeze(1).repeat(1, Path2).flatten().detach().cpu()
            _X = self.pack_lstm(X.flatten(0, 1), L, Path1).view(X.shape)
        else:
            _X = self.lstm(X.flatten(0, 1))[0].view(X.shape)
        X = (X + self.norm(self.cond(self.proj(_X), C))) * math.sqrt(0.5)
        return X

    def pack_lstm(self, X, lengths, total_length):
        # Packing
        X = nn.utils.rnn.pack_padded_sequence(
            X, lengths, batch_first=True, enforce_sorted=False)
        # lstm
        X = self.lstm(X)[0]
        # Unpacking
        X = nn.utils.rnn.pad_packed_sequence(
            X, total_length=total_length, batch_first=True)[0]
        return X

    def cond(self, X, C=None):
        if C is None:
            return X
        elif self.cond_in:
            Batch = C.size(0)
            Ca = self.cond_a(C).view(
                Batch, -1, 1, 1).exp() if self.cond_a else 1.0
            Cb = self.cond_b(C).view(Batch, -1, 1, 1) if self.cond_b else 0.0
            X = self.cond_in(X.transpose(1, 3))
            X = self.cond_nl(Ca * X + Cb)
            return X.transpose(1, 3).contiguous()
        else:
            Batch = C.size(0)
            Ca = self.cond_a(C).view(
                Batch, 1, 1, -1).exp() if self.cond_a else 1.0
            Cb = self.cond_b(C).view(Batch, 1, 1, -1) if self.cond_b else 0.0
            return self.cond_nl(Ca * X + Cb)


class ConvBlock(nn.Module):
    def __init__(self,
                 num_hidden,
                 num_condition=0,
                 num_condition_group=1,
                 dropout=0.1,
                 kernel_size=3,
                 use_affine=True
                 ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            num_hidden, num_hidden, kernel_size, padding=(kernel_size-1)//2
        )
        self.proj = nn.Sequential(
            nn.GroupNorm(1, num_hidden),
            nn.PReLU(num_parameters=num_hidden, init=0.2),
            nn.Conv1d(
                num_hidden, num_hidden, 1,
            ),
            nn.GroupNorm(1, num_hidden),
            nn.Dropout(p=dropout)
        )
        assert num_hidden % num_condition_group == 0

        if num_condition > 0:
            self.cond_b = nn.Linear(
                num_condition, num_hidden//num_condition_group)
        else:
            self.cond_b = None

        if num_condition > 0 and use_affine:
            self.cond_a = nn.Linear(
                num_condition, num_hidden//num_condition_group)
        else:
            self.cond_a = None

    def forward(self, X, C=None, L=None):
        ''' Convolutional block for dual-path model
        Input: 
            X: Size of (Batch, Path1, Path2, Dim)
            C: Size(Batch, Group, C_Dim) or Size(Batch, C_Dim)
            L: Size of (Batch, Path1)
        Output: 
            X: Size of (Batch, Path2, Path1, Dim)
        '''
        Batch, Path1, Path2, Dim = X.shape
        X = X.permute(0, 2, 3, 1).contiguous()
        _X = self.conv(X.flatten(0, 1)).view(X.shape)
        Ca = self.cond_a(C).view(Batch, 1, -1, 1).exp() if self.cond_a else 1.0
        Cb = self.cond_b(C).view(Batch, 1, -1, 1) if self.cond_b else 0.0
        X = (X + self.proj(Ca * _X + Cb)) * math.sqrt(0.5)
        X = X.transpose(2, 3).contiguous()
        return X


class GatedTanhUnit(nn.Module):
    def __init__(self, num_hidden, dim=1):
        super(GatedTanhUnit, self).__init__()
        self.num_hidden = num_hidden
        self.dim = dim

    def forward(self, input):
        assert input.size(self.dim) == self.num_hidden*2
        inputs = input.split(self.num_hidden, dim=self.dim)
        return torch.tanh(inputs[0]) * torch.sigmoid(inputs[1])


class DualPathNNet(nn.Module):
    def __init__(self,
                 num_input,
                 num_hidden,
                 num_output,
                 num_condition=0,
                 num_group=1,
                 chunk_size=1,
                 hop_size=1,
                 module_path1=['blstm'],
                 module_path2=['blstm'],
                 dropout=0.1,
                 fold=True,
                 ):
        super(DualPathNNet, self).__init__()

        assert len(module_path1) == len(module_path2)
        assert num_hidden % num_group == 0
        self.module_path1 = module_path1
        self.module_path2 = module_path2
        self.num_layers = len(module_path1)
        self.num_hidden = num_hidden
        self.num_condition = num_condition
        self.num_group = num_group
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.fold_param = dict(
            kernel_size=(chunk_size, 1),
            padding=((chunk_size-1)//2, 0),
            stride=(hop_size, 1),
        )

        self.block_path1 = nn.ModuleList()
        self.block_path2 = nn.ModuleList()
        for module_name1, module_name2 in zip(module_path1, module_path2):
            self.block_path1.append(self.make_block(module_name1))
            self.block_path2.append(self.make_block(module_name2))

        self.group_net = nn.Sequential(
            nn.PReLU(num_parameters=1, init=0.2),
            nn.Conv2d(num_hidden, num_hidden, 1),
        )

        num_hidden = num_hidden // num_group
        self.pre_net = nn.Sequential(
            nn.GroupNorm(1, num_input),
            nn.Conv1d(num_input, num_hidden, 1),
        )
        self.post_net = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, 1),
            nn.GroupNorm(2, num_hidden*2),
            GatedTanhUnit(num_hidden, dim=1),
            nn.Conv1d(num_hidden, num_output, 1, bias=False),
        )

    def make_block(self, module_name):
        params = {
            'num_condition': self.num_condition,
            'num_condition_group': self.num_group,
            'dropout': self.dropout
        }
        if module_name == 'lstm':
            return LSTMBlock(self.num_hidden, **params)
        elif module_name == 'blstm':
            return LSTMBlock(self.num_hidden, bidirectional=True, **params)
        elif module_name == 'conv':
            return ConvBlock(self.num_hidden, kernel_size=3, **params)

    def forward(self, X, C=None, L1=None, L2=None, output_each_layer=False):
        '''
        Input:
            X: Size(Batch*Group, Dim, Time)
            C: Size(Batch*Group, C_Dim)
            L1: Size of (Batch, Path1), length of 
            L2: Size of (Batch, Path2)
        Output:
            X: Size(Batch*Group, Dim, Time)
        '''
        X = self.pre_net(X)
        Batch, Dim, Time = X.shape
        Chunk = self.chunk_size
        Group = self.num_group
        Batch = Batch // Group

        X = F.unfold(X.unsqueeze(-1), **self.fold_param)
        X = X.view(Batch, Group*Dim, Chunk, -
                   1).permute(0, 2, 3, 1).contiguous()
        if C is not None:
            C = C.view(Batch, Group, -1)

        if output_each_layer:
            Xs = []
            for i in range(self.num_layers):
                X = self.block_path1[i](X, C=C, L=L1)
                X = self.block_path2[i](X, C=C, L=L2)

                _X = X.permute(0, 3, 1, 2).contiguous()
                _X = self.group_net(_X)
                _X = _X.view(Batch*Group, Dim*Chunk, -1)
                _X = F.fold(_X, (Time, 1), **self.fold_param).squeeze(-1)

                Xs.append(self.post_net(_X))
            return Xs
        else:
            for i in range(self.num_layers):
                X = self.block_path1[i](X, C=C, L=L1)
                X = self.block_path2[i](X, C=C, L=L2)

            X = X.permute(0, 3, 1, 2).contiguous()
            X = self.group_net(X)
            X = X.view(Batch*Group, Dim*Chunk, -1)
            X = F.fold(X, (Time, 1), **self.fold_param).squeeze(-1)

            return self.post_net(X)

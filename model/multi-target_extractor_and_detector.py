#!/usr/bin/env python3

# Copyright 2022 IIS, Academia Sinica. (author: Chin-Yi Cheng)
#

import torch
import torch.nn as nn

from .resnetASP import ResNet
from .dual_path_nnet import LSTMBlock, ConvBlock, GatedTanhUnit


class TimeSpeakerContextualizer(nn.Module):
    def __init__(self,
                 num_input,
                 num_hidden,
                 num_condition=0,
                 num_group=1,
                 module_path1=['blstm'],
                 module_path2=['blstm'],
                 dropout=0.1,
                 ):
        super(TimeSpeakerContextualizer, self).__init__()

        assert len(module_path1) == len(module_path2)
        assert num_hidden % num_group == 0
        self.module_path1 = module_path1
        self.module_path2 = module_path2
        self.num_layers = len(module_path1)
        self.num_hidden = num_hidden
        self.num_condition = num_condition
        self.num_group = num_group
        self.dropout = dropout

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
            nn.Conv2d(num_input, num_hidden, 1),
        )
        self.post_net = nn.Sequential(
            nn.Conv1d(num_hidden, num_hidden*2, 1),
            nn.GroupNorm(2, num_hidden*2),
            GatedTanhUnit(num_hidden, dim=1),
            nn.Conv1d(num_hidden, 1, 1, bias=False),
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

    def forward(self, X, C=None, L1=None, L2=None):
        '''
        Input:
            X: Size(Batch, Group, Dim, Time)
            C: Size(Batch, Group, C_Dim)
        Output:
            X: Size(Batch, Group, Time)
        '''
        Batch, Group, Dim, Time = X.shape
        X = self.pre_net(X.transpose(1, 2))

        X = X.permute(0, 2, 3, 1).contiguous()   # Batch, Group, Time, Dim
        if C is not None:
            C = C.view(Batch, Group, -1)

        for i in range(self.num_layers):
            X = self.block_path1[i](X, C=C, L=L1)
            X = self.block_path2[i](X, C=C, L=L2)

        X = X.transpose(1, 3).contiguous()   # Batch, Dim, Time, Group
        X = self.group_net(X)

        X = X.permute(0, 3, 1, 2).contiguous()  # Batch, Dim, Time, Group
        return self.post_net(X.flatten(0, 1)).view(Batch, Group, Time)


class CNN(nn.Module):
    def __init__(self,
                 num_input=40,
                 num_hidden=128,
                 ):
        super(CNN, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(num_input, num_hidden, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden),
            nn.Conv1d(num_hidden, num_hidden, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden),
            nn.Conv1d(num_hidden, num_hidden, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden),
            nn.Conv1d(num_hidden, num_hidden, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden),
        )

    def forward(self, input):
        return self.net(input)


class RNN(nn.Module):
    def __init__(self,
                 num_input=256,
                 num_hidden=128,
                 num_layer=2,
                 bidirectional=True,
                 dropout=0.1
                 ):
        super(RNN, self).__init__()

        self.linear = nn.Linear(num_hidden, num_hidden)
        self.rnn = nn.LSTM(
            num_input, num_hidden//2 if bidirectional else num_hidden,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, input):
        input = self.rnn(input.transpose(1, 2))[0]
        input = self.linear(input).transpose(1, 2)
        return input.contiguous()


class RNNVAD(nn.Module):
    def __init__(self,
                 num_group=4,
                 num_hidden=128,
                 num_layer=2,
                 bidirectional=True,
                 dropout=0.1
                 ):
        super(RNNVAD, self).__init__()

        self.proj = nn.Linear(
            num_hidden*num_group,
            num_hidden*num_group
        )
        self.vad = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
        )
        self.rnn = nn.LSTM(
            num_hidden*num_group,
            num_hidden*num_group//2 if bidirectional else num_hidden,
            num_layers=num_layer,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(self, input):
        Batch, Spk, Dim, Time = input.shape
        input = self.rnn(input.flatten(1, 2).transpose(1, 2))[0]
        input = self.proj(input).view(Batch, Time, Spk, -1)
        input = self.vad(input).transpose(1, 2).contiguous()
        return input.view(Batch, Spk, Time)


class Model(nn.Module):
    """
    Main model of MTFAD zvec
    """

    def __init__(self, arch):
        super(Model, self).__init__()

        self.pre_net = CNN(**arch['pre_net'])
        self.spk_encoder = RNN(**arch['spk_net'])
        self.resnet = ResNet()
        self.linearPreRes = nn.Linear(40, 513)
        self.linearPostRes = nn.Linear(256, 128)
        self.vad_decoder = TimeSpeakerContextualizer(**arch['TSCont'])

    def forward(self, X, E, num_spks=None):
        '''
        Input:
            X: Features, i.e. MFCC, Size(Batch, Dim, Time)
            E: Speaker Features, i.e. 500 frames of speaker, Size(Batch, Time , E_Dim, Spk)
            num_spks: Numbers of speaker, for training, Size(Batch)
        Output:
            Y: VAD result, Size(Batch, Spk, Time)
        '''
        E_batch, _, E_Dim, E_spk = E.shape
        E = E.permute(0, 3, 1, 2).flatten(0, 1)  # 32,500,40
        E = self.linearPreRes(E).permute(0, 2, 1)  # 32,513,500
        E = self.resnet(E).view(E_batch, E_spk, -1)  # 8,4,256
        E = self.linearPostRes(E)
        X_encode = self.pre_net(X)

        # Randomly permute embeddings along Spk axis
        if self.training:
            E_randperm, reverse_idx = random_permute(
                E, dim=1, lengths=num_spks, return_reverse_idx=True)
        else:
            E_randperm = E.contiguous()

        # Concatenate frame-based embeddings and speaker embeddings
        Time, Spk = X.size(-1), E.size(1)
        X_E = torch.cat([
            X_encode.unsqueeze(1).repeat(1, Spk, 1, 1),
            E_randperm.unsqueeze(-1).repeat(1, 1, 1, Time)
        ], dim=2)

        Batch, Spk, Dim, Time = X_E.shape

        # Do target speaker VAD
        X_spk = self.spk_encoder(X_E.flatten(0, 1)).view(Batch, Spk, -1, Time)
        if isinstance(self.vad_decoder, TimeSpeakerContextualizer):
            X_vad = self.vad_decoder(X_spk, C=None, L1=num_spks)
        else:
            X_vad = self.vad_decoder(X_spk)

        # Reverse VAD results generated form permuted embeddings along Spk axis
        if self.training:
            X_vad = reverse_permuted(X_vad, reverse_idx, dim=1)
        return X_vad


def random_permute(features, dim=-1, lengths=None, return_reverse_idx=False):
    if lengths is None:
        rand_idx = torch.randperm(
            features.size(dim),
            dtype=torch.long, device=features.device
        )
        features = features.index_select(dim=dim, index=rand_idx)
    else:
        assert dim == 1
        features_shape = features.shape
        Batch = features_shape[0]
        total_length = features_shape[dim]
        device = features.device

        rand_idx = []
        for length in lengths:
            rand_idx_ = torch.arange(total_length,
                                     dtype=torch.long, device=device)
            rand_idx_[:length] = torch.randperm(length,
                                                dtype=torch.long, device=device)

            rand_idx.append(rand_idx_.unsqueeze(0))
        rand_idx = torch.cat(rand_idx, dim=0)

        rand_idx_ = rand_idx + \
            torch.arange(0, Batch*total_length, total_length,
                         device=device).unsqueeze(1)
        features = features.flatten(0, 1).index_select(
            dim=0, index=rand_idx_.flatten().detach())
        features = features.view(features_shape)

    if return_reverse_idx:
        if lengths is None:
            reverse_idx = torch.argsort(rand_idx)
        else:
            reverse_idx = torch.argsort(rand_idx, dim=1)
        return features, reverse_idx
    else:
        return features


def reverse_permuted(features, reverse_idx, dim=-1):
    if reverse_idx.ndim == 1:
        return features.index_select(dim=dim, index=reverse_idx)
    elif reverse_idx.ndim == 2:
        assert dim == 1
        features_shape = features.shape
        Batch = features_shape[0]
        total_length = features_shape[dim]
        device = features.device
        reverse_idx_ = reverse_idx + \
            torch.arange(0, Batch*total_length, total_length,
                         device=device).unsqueeze(1)
        features = features.flatten(0, 1).index_select(
            dim=0, index=reverse_idx_.flatten().detach())
        features = features.view(features_shape)
        return features


def get_mask_from_lengths(lengths, max_length=None):
    max_len = torch.max(lengths).item() if max_length is None else max_length
    ids = torch.arange(0, max_len, dtype=torch.long, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask

#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from configargparse import ArgParser
from slowfast.models.nonlocal_helper import Nonlocal
import numpy as np

class HDRModule(nn.Module):
    def __init__(self, in_channels, num_frames=4, intra=True, embed=True, inter_channels=256):
        super(HDRModule, self).__init__()
        print(f"building HDR Module use_intra: {intra}, use_embed: {embed}")

        self.use_intra = intra
        self.use_embed = embed
        
        self.inter = Nonlocal(in_channels, inter_channels)

        if self.use_intra:
            self.intra = Nonlocal(in_channels, inter_channels)
        if self.use_embed:
            self.embedding = nn.Embedding(10, in_channels)
        self.num_frames = num_frames

    def forward(self, x, masks):
        # x: [q1, choices, q2, padding]

        N, C, T, H, W = x.shape
        F = self.num_frames
        M = T // F # M = max_length + 2
        assert M == 9, f"M = {M}, F = {F}"

        # plus embedding
        x = x.transpose(1, 2) # N, T, C, H, W
        y = x.reshape(N, M, F, C, H, W)

        if self.use_embed:
            embedding = self.embedding(masks).view(N, M, 1, C, 1, 1)
            y = y + embedding

        # intra
        intra_input = y.reshape(N * M, F, C, H, W).transpose(1, 2)
        if self.use_intra:
            intra_res = self.intra(intra_input)
        else:
            intra_res = intra_input

        # N, C, M, H, W
        inter_input = intra_res.mean(2).reshape(N, M, C, H, W)
        inter_input = inter_input.transpose(1, 2)
        inter_res = self.inter(inter_input)

        inter_res = inter_res.transpose(1, 2).reshape(N, M, 1, C, H, W)
        res = inter_res
        
        if self.use_intra:
            intra_res = intra_res.transpose(1, 2).reshape(N, M, F, C, H, W)
            res = res + intra_res
        else:
            res = res.expand(-1, -1, F, -1, -1, -1)
        res = res.reshape(N, M * F, C, H, W)
        x = (x + res).transpose(1, 2)
        return x
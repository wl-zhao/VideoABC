#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoABCReasonLoss():
    def __init__(self, **kwargs):
        pass

    def __call__(self, outputs, labels):
        ans_label, type_label, pos_label = labels
        N, K, _ = outputs.shape

        ans = outputs[:, :, 0] # (N, K)
        loss = {}
        loss['ans'] = F.binary_cross_entropy_with_logits(ans, ans_label)

        wrong_outputs = outputs[:, 1:]
        loss['wrong_type'] = 0
        loss['wrong_pos'] = 0
        for k in range(K - 1):
            wrong_type = wrong_outputs[:, k, 1: 5]
            loss['wrong_type'] += F.cross_entropy(wrong_type, type_label[:, k])

            loss_wrong_pos = 0
            for n in range(N):
                output_pos = wrong_outputs[n, k, 5:].view(5, -1)
                idx = type_label[n, k]
                if idx == 3: # swap
                    loss_wrong_pos += F.cross_entropy(output_pos[-2:], pos_label[n, :, k])
                else:
                    loss_wrong_pos += F.cross_entropy(output_pos[idx: idx + 1], pos_label[n, :1, k])
            loss_wrong_pos /= N
            loss['wrong_pos'] += loss_wrong_pos

        loss['wrong_type'] /= (K - 1)
        loss['wrong_pos'] /= (K - 1)
        loss['total'] = sum(loss.values())
        return loss

_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_logit": nn.BCEWithLogitsLoss,
    "videoabc_reason_loss": VideoABCReasonLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Functions for computing metrics."""

import torch


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.

    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.

    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [top_max_k_correct[:k, :].float().sum() for k in ks]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

def reasoning_accuracies2(outputs, labels):
    ans_label, type_label, pos_label = labels

    N, K, _ = outputs.shape

    ans = torch.sigmoid(outputs[:, :, 0])
    acc = {}
    ans_correct = ((ans > 0.5) == ans_label).float() * 100
    acc['ans'] = (ans_correct[:, 0].mean() + ans_correct[:, 1:].mean()) / 2

    wrong_outputs = outputs[:, 1:]
    acc['type'] = 0
    acc['reason'] = 0
    type_sums = [0] * 4
    type_corrects = [0] * 4
    for k in range(K - 1):
        wrong_type = wrong_outputs[:, k, 1: 5]

        type_correct = accuracy(wrong_type, type_label[:, k], True)[0]
        acc['type'] += type_correct.float().mean() * 100

        pos_corrects = []
        for n in range(N):
            output_pos = wrong_outputs[n, k, 5:].view(5, -1)
            idx = type_label[n, k]
            type_sums[idx] += 1
            if idx == 3: # swap
                pos_correct = accuracy(output_pos[-2:], pos_label[n, :, k], True)[0]
                pos_correct = torch.logical_and(pos_correct[0], pos_correct[1])
            else:
                pos_correct = accuracy(output_pos[idx: idx + 1], pos_label[n, :1, k], True)[0, 0]
            type_corrects[idx] += pos_correct * type_correct[n]

            pos_corrects.append(pos_correct)
        pos_correct = torch.stack(pos_corrects)

        reason_correct = torch.logical_and(type_correct, pos_correct)
        acc['reason'] += reason_correct.float().mean() * 100

    acc['type'] /= (K - 1)
    acc['reason'] /= (K - 1)
    types = ['remove', 'insert', 'replace', 'swap']
    for i in range(4):
        t = types[i]
        acc[f'reason/{t}'] = type_corrects[i] / type_sums[i]
    return type_sums, type_corrects

def reasoning_accuracies1(outputs, labels):
    ans_label, type_label, pos_label = labels

    N, K, _ = outputs.shape

    ans = torch.sigmoid(outputs[:, :, 0])
    acc = {}
    ans_correct = ((ans > 0.5) == ans_label).float() * 100
    acc['ans'] = (ans_correct[:, 0].mean() + ans_correct[:, 1:].mean()) / 2

    wrong_outputs = outputs[:, 1:]
    acc['type'] = 0
    acc['reason'] = 0
    for k in range(K - 1):
        wrong_type = wrong_outputs[:, k, 1: 5]

        type_correct = accuracy(wrong_type, type_label[:, k], True)[0]
        acc['type'] += type_correct.float().mean() * 100

        pos_corrects = []
        for n in range(N):
            output_pos = wrong_outputs[n, k, 5:].view(5, -1)
            idx = type_label[n, k]
            if idx == 3: # swap
                pos_correct = accuracy(output_pos[-2:], pos_label[n, :, k], True)[0]
                pos_correct = torch.logical_and(pos_correct[0], pos_correct[1])
            else:
                pos_correct = accuracy(output_pos[idx: idx + 1], pos_label[n, :1, k], True)[0, 0]
            pos_corrects.append(pos_correct)
        pos_correct = torch.stack(pos_corrects)

        reason_correct = torch.logical_and(type_correct, pos_correct)
        acc['reason'] += reason_correct.float().mean() * 100

    acc['type'] /= (K - 1)
    acc['reason'] /= (K - 1)
    return acc

def reasoning_accuracies(outputs, labels):
    ans_label, wrong_indices, type_label, pos_label, pos_mask = labels

    N, K, P = outputs.shape
    max_length = (P - 5) // 2

    ans = outputs[:, :, 0]
    acc = {}
    acc['ans'] = accuracy(ans, ans_label)[0]

    wrong_indices = wrong_indices.unsqueeze(-1).expand(N, K - 1, outputs.shape[-1])
    wrong_outputs = torch.gather(outputs, dim=1, index=wrong_indices)
    acc['type'] = 0
    acc['reason'] = 0
    for k in range(K - 1):
        wrong_type = wrong_outputs[:, k, 1: 5]

        type_correct = accuracy(wrong_type, type_label[:, k], True)
        acc['type'] += type_correct.float().mean() * 100

        wrong_pos1 = wrong_outputs[:, k, 5: 5 + max_length]
        wrong_pos2 = wrong_outputs[:, k, 5 + max_length: ]

        pos1_correct = accuracy(wrong_pos1, pos_label[:, 0, k], True)
        pos2_correct = accuracy(wrong_pos2, pos_label[:, 1, k], True)
        pos2_correct = torch.logical_or(pos2_correct, torch.logical_not(pos_mask[:, k]))
        pos_correct = torch.logical_and(pos1_correct, pos2_correct)

        reason_correct = torch.logical_and(type_correct, pos_correct)
        acc['reason'] += reason_correct.float().mean() * 100

    acc['type'] /= (K - 1)
    acc['reason'] /= (K - 1)
    return acc

@torch.no_grad()
def accuracy(output, target, reture_array=False, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    if reture_array:
        return correct

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

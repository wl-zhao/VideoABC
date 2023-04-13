#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms
from fvcore.common.file_io import PathManager
import json
import numpy as np
from PIL import Image
from glob import glob
import pandas as pd
from torchvideotransforms import video_transforms, volume_transforms

import slowfast.utils.logging as logging

from . import decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)

def smart_load(obj):
    return json.loads(obj) if isinstance(obj, str) else obj

WRONG_TYPES = ['remove', 'insert', 'replace', 'swap']
@DATASET_REGISTRY.register()
class Videoabc(torch.utils.data.Dataset):
    """
    VideoABC Dataset
    """

    def __init__(self, cfg, mode):
        self.cfg = cfg

        if mode == 'val':
            mode = 'test'
        self.mode = mode

        self.df = pd.read_csv(f"data_gen/anno_{cfg.DATA.SETTING}.csv")
        self.metadata = json.load(open(f"{cfg.DATA.METADATA}/{cfg.DATA.SETTING}/{mode}.json"))
        self.num_frames = cfg.DATA.CLIP_LEN
        self.root = cfg.DATA.PATH_TO_DATA_DIR

        setting = cfg.DATA.SETTING
        if 'long' in setting:
            self.max_length = int(setting[-1]) + 1
        else: self.max_length = 1

        cfg.MODEL.NUM_CLASSES = 1 + 4 + 5 * self.max_length

        self.mode = mode
        self._init_transforms()

    def _init_transforms(self):
        tail_transforms = [
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        ]

        if self.mode == 'train':
            train_transform = [
                video_transforms.Resize(128),
                video_transforms.RandomCrop(112),
                video_transforms.RandomHorizontalFlip(),
            ]
            self.transforms = video_transforms.Compose(train_transform + tail_transforms)
        else:
            val_transforms = [
                video_transforms.Resize(128),
                video_transforms.CenterCrop(112),
            ]
            self.transforms = video_transforms.Compose(val_transforms + tail_transforms)


    def _gen_wrong_choices(self, qa):
        correct = qa['correct']
        video_name = qa['video_name']
        row = self.df.loc[self.df.video_name == video_name].iloc[0]
        action_ids = smart_load(row["action_ids"])
        clip = qa['clip']

        wrong_type_id = np.random.randint(4)
        correct_steps = correct['steps']
        steps = correct_steps.copy()
        if wrong_type_id == 0: # remove
            idx = np.random.randint(len(correct_steps))
            steps.pop(idx)
            wrong = {
                'steps': steps,
                'correct': False,
                'wrong_type': 'remove',
                'wrong_pos': [int(idx)],
            }
        elif wrong_type_id == 1: # insert
            idx = np.random.randint(len(action_ids))
            new = '{}/{}'.format(video_name, idx)
            insert_step = np.random.randint(len(steps))
            steps.insert(insert_step, new)
            wrong = {
                'steps': steps,
                'correct': False,
                'wrong_type': 'insert',
                'wrong_pos': [int(insert_step)],
            }
        elif wrong_type_id == 2: # replace
            i1 = np.random.randint(len(steps))
            i2s = [i for i in range(len(action_ids)) if action_ids[i] != action_ids[i1 + clip[0]]]

            if len(i2s) != 0: # replace same
                i2 = np.random.choice(i2s)
                steps[i1] = '{}/{}'.format(video_name, i2)
            else: # replace other
                item = self.df.loc[(self.df.video_class == row['video_class']) & (self.df.video_name != row['video_name'])].sample(1).iloc[0]
                i2 = np.random.randint(item['num_actions'])
                steps[i1] = '{}/{}'.format(item['video_name'], i2)
            wrong = {
                'steps': steps,
                'correct': False,
                'wrong_type': 'replace',
                'wrong_pos': [int(i1)],
            }
        else: # swap
            i1 = np.random.randint(len(steps) - 1)
            i2 = np.random.randint(i1 + 1, len(steps))
            steps[i1], steps[i2] = steps[i2], steps[i1]
            wrong = {
                'steps': steps,
                'correct': False,
                'wrong_type': 'swap',
                'wrong_pos': sorted([int(i1), int(i2)]),
            }
        qa['choices'] = [correct, wrong]
        return qa

    def __getitem__(self, index):
        qa = self.metadata[index]
        if self.mode == "train":
            qa = self._gen_wrong_choices(qa)
            del qa['correct']

        label = self._gen_labels(qa)

        frames = []

        # add questions
        frames.extend([
            Image.open(f'{self.root}/{q}') for q in qa['question']
        ])

        for choice in qa['choices']:
            for step in choice['steps']:
                frames.extend(self._load_step(step))

        frames = self.transforms(frames)

        questions = frames[:, :2]

        start = 2

        slow_pathways = []
        fast_pathways = []

        q1 = questions[:, :1].expand(-1, 8, -1, -1)
        q2 = questions[:, 1:].expand(-1, 8, -1, -1)
        masks = []
        for choice in qa['choices']:
            num_steps = len(choice['steps'])
            end = start + num_steps * self.num_frames
            choice = frames[:, start: end] # C, T, H, W
            C, T, H, W = choice.shape
            padding_T = 56 - num_steps * self.num_frames
            padding = torch.zeros(C, padding_T, H, W)

            mask = torch.arange(0, self.max_length + 2) + 2
            mask[0] = 1
            mask[num_steps + 1] = 2
            mask[num_steps + 2:] = 0
            masks.append(mask)

            fast_pathway = torch.cat([q1, choice, q2, padding], dim=1)
            slow_pathway, fast_pathway = utils.pack_pathway_output(self.cfg, fast_pathway)

            slow_pathways.append(slow_pathway)
            fast_pathways.append(fast_pathway)

            start = end

        slow_pathways = torch.stack(slow_pathways)
        if True:# fast_pathways[0]:
            fast_pathways = torch.stack(fast_pathways)
        else:
            fast_pathways = None
        masks = torch.stack(masks)
        frames = [slow_pathways, fast_pathways, masks]

        return frames, label, index, {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.metadata)

    def _load_step(self, step):
        """load self.num_frame frames from the step

        Returns:
            list of torch.Tensor: (3, H, W)
        """
        index = (16 - self.num_frames) // 2
        paths = sorted(glob(f'{self.root}/{step}/*.jpg'))[index: index + self.num_frames]
        images = [Image.open(path) for path in paths]
        return images

    def _gen_labels(self, qa):
        type_label = []
        pos_label1 = []
        pos_label2 = []

        for choice in qa['choices']:
            if choice['correct']:
                continue
            type_label.append(WRONG_TYPES.index(choice['wrong_type']))
            wrong_pos = choice['wrong_pos']
            wrong_pos = sorted(wrong_pos)
            pos_label1.append(wrong_pos[0])
            if len(wrong_pos) == 2:
                pos_label2.append(wrong_pos[1])
            else:
                pos_label2.append(0)

        ans_label = torch.FloatTensor([c['correct'] for c in qa['choices']])
        type_label = torch.LongTensor(type_label)
        pos_label = torch.LongTensor([pos_label1, pos_label2])
        return ans_label, type_label, pos_label

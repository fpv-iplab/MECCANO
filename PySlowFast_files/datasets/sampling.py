#!/usr/bin/env python3

import math
import numpy as np
import random
import torch
import torchvision.io as io


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval. If the number of frames is < num_samples, duplicate frames.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """


    length = (end_idx - start_idx)+1
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, end_idx).long()
    index = index-start_idx

    out_frames = torch.index_select(frames, 0, index)

    return out_frames

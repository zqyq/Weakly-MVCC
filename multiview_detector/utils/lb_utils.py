import torch
import numpy as np


def concat_b(data_b: list, b=None):
    if b is None:
        b = len(data_b)
    return torch.cat(data_b).reshape((b,) + data_b[0].shape)
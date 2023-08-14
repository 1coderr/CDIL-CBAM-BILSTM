# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   数据加载
# ----------------------------------------------------#
import numpy as np
import torch
from datasets import Dataset


class MyDataset(Dataset):
    def __init__(self, signals, targets):
        self.targets = targets
        self.signals=signals[:,np.newaxis].astype(np.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        return self.signals[index],self.targets[index]


def list2tensor(li):
    return torch.tensor(np.array(li))

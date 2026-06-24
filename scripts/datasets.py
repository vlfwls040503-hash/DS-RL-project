# -*- coding: utf-8 -*-
"""Dataset / split / scaling utilities for BC training."""
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from common import CACHE, FEATURES, ACTIONS, VAL_SUBJECTS, TEST_SUBJECTS


def load_cache():
    d = np.load(os.path.join(CACHE, "dataset.npz"), allow_pickle=True)
    return (d["X"].astype("float32"), d["Y"].astype("float32"),
            d["run_id"].astype("int64"), d["subject"].astype("int64"),
            d["scenario"].astype("int64"))


def split_masks(subject):
    test = np.isin(subject, TEST_SUBJECTS)
    val = np.isin(subject, VAL_SUBJECTS)
    train = ~(test | val)
    return train, val, test


class Scaler:
    """z-score scaler fit on a subset; stored as float32 tensors."""
    def __init__(self, mean, std):
        self.mean = mean.astype("float32")
        self.std = np.where(std < 1e-8, 1.0, std).astype("float32")

    @classmethod
    def fit(cls, arr):
        return cls(arr.mean(axis=0), arr.std(axis=0))

    def transform(self, arr):
        return (arr - self.mean) / self.std

    def inverse(self, arr):
        return arr * self.std + self.mean

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.array(d["mean"], dtype="float32"), np.array(d["std"], dtype="float32"))


class FrameDataset(Dataset):
    """Single-frame (state -> action)."""
    def __init__(self, X, Y, idx):
        self.X = torch.from_numpy(X[idx])
        self.Y = torch.from_numpy(Y[idx])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


class SeqDataset(Dataset):
    """Sliding window (L frames, stride s) -> action at last frame.
    Windows never cross run boundaries. X is shared (not copied)."""
    def __init__(self, X, Y, run_id, mask, L=24, stride=2):
        self.X = torch.from_numpy(X)            # full array, shared
        self.Y = torch.from_numpy(Y)
        self.L, self.stride = L, stride
        span = (L - 1) * stride                 # frames needed before end position
        e = np.where(mask)[0]
        s = e - span
        ok = s >= 0
        e, s = e[ok], s[ok]
        same = run_id[s] == run_id[e]            # whole window inside one run
        self.ends = e[same].astype("int64")

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        e = self.ends[i]
        s = e - (self.L - 1) * self.stride
        win = self.X[s:e + 1:self.stride]        # (L, F)
        return win, self.Y[e]

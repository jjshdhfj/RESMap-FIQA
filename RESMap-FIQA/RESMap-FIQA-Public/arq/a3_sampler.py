# -*- coding: utf-8 -*-
from __future__ import annotations
# Exact definitions extracted automatically from: utils.py

from torch.utils.data import Sampler
from typing import Iterator
import torch
import random
import numpy as np

def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class EpochShuffleSampler(Sampler):
    """Deterministic full-dataset permutation keyed by seed and epoch."""

    def __init__(self, data_source, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            len(self.data_source),
            generator=generator,
        ).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"epoch must be a non-negative int, got {epoch!r}")
        self.epoch = epoch


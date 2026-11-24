# -*- coding: utf-8 -*-
from __future__ import annotations
# Exact definitions extracted automatically from: restoration_evidence.py

from torch import Tensor
from typing import Optional
from .data import _resize
from typing import Tuple
import torch

def apply_joint_geometry(
    rgb: Tensor, u_map: Tensor, response: Tensor, mask: Tensor, *,
    hflip: bool = False,
    crop: Optional[Tuple[int, int, int, int]] = None,
    resize: Optional[Tuple[int, int]] = None,
):
    tensors = (rgb, u_map, response, mask)
    spatial = tuple(rgb.shape[-2:])
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise AssertionError("rgb must be [3,H,W]")
    if any(item.ndim != 3 or item.shape[0] != 1 for item in tensors[1:]):
        raise AssertionError("U/Response/mask must each be [1,H,W]")
    if any(tuple(item.shape[-2:]) != spatial for item in tensors):
        raise AssertionError("All paired inputs must share their source geometry")
    if crop is not None:
        top, left, height, width = map(int, crop)
        if min(top, left, height, width) < 0 or height == 0 or width == 0:
            raise ValueError(f"invalid paired crop: {crop}")
        if top + height > spatial[0] or left + width > spatial[1]:
            raise ValueError(f"crop outside paired source: {crop}, {spatial}")
        tensors = tuple(item[:, top:top + height, left:left + width] for item in tensors)
    if hflip:
        tensors = tuple(torch.flip(item, dims=(-1,)) for item in tensors)
    if resize is not None:
        rgb, u_map, response, mask = tensors
        tensors = (
            _resize(rgb, resize, "bilinear"),
            _resize(u_map, resize, "bilinear"),
            _resize(response, resize, "bilinear"),
            _resize(mask, resize, "nearest"),
        )
    return tuple(item.contiguous() for item in tensors)


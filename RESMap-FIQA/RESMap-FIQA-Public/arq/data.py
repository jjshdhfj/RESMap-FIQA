"""Data loading for the four-task ARQ model."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .constants import TASK_NAMES


RISK_SIZE = (512, 512)
RETFOUND_SIZE = (384, 384)
LABEL_COLUMNS = {
    "clarity": "清晰度",
    "exposure": "曝光",
    "artifact": "伪影",
    "completeness": "关键区域完整度",
}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _shape_error(
    label: str,
    actual: Sequence[int],
    expected: Sequence[int],
    name: str,
    path: str,
) -> AssertionError:
    return AssertionError(
        f"{label} shape mismatch for sample name={name!r}, path={path!r}: "
        f"actual={tuple(actual)}, expected={tuple(expected)}"
    )


def _assert_shape(
    tensor: Tensor,
    expected: Sequence[int],
    label: str,
    name: str,
    path: str,
) -> None:
    if tuple(tensor.shape) != tuple(expected):
        raise _shape_error(label, tensor.shape, expected, name, path)


def _build_stem_index(
    roots: Iterable[Path],
    extensions: Sequence[str],
    label: str,
) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    allowed = {extension.lower() for extension in extensions}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Required {label} directory does not exist: {root}")
        entries = sorted(os.scandir(str(root)), key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            if not entry.is_file() or path.suffix.lower() not in allowed:
                continue
            stem = path.stem
            if stem in index:
                raise ValueError(
                    f"Duplicate {label} stem {stem!r}: "
                    f"{index[stem]} and {path}"
                )
            index[stem] = path
    return index


def _read_rgb(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _read_mask(path: Path) -> Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.uint8, copy=True)
    return torch.from_numpy(array).unsqueeze(0).float().div_(255.0)


def _read_umap(path: Path, name: str) -> Tensor:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array.transpose(2, 0, 1)
    if array.ndim != 3 or array.shape[0] != 1:
        raise _shape_error(
            "umap source",
            array.shape,
            (1, "H", "W"),
            name,
            str(path),
        )
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(
            f"umap contains non-finite values for sample name={name!r}, "
            f"path={str(path)!r}"
        )
    return torch.from_numpy(np.ascontiguousarray(array))


def _resize(tensor: Tensor, size: Tuple[int, int], mode: str) -> Tensor:
    kwargs = {"size": size, "mode": mode}
    if mode == "bilinear":
        kwargs["align_corners"] = False
    return F.interpolate(tensor.unsqueeze(0), **kwargs).squeeze(0)


def apply_paired_geometry(
    rgb: Tensor,
    u_map: Tensor,
    *,
    hflip: bool = False,
    crop: Optional[Tuple[int, int, int, int]] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Tensor]:
    """Apply one sampled geometry to RGB and its continuous U-map evidence."""
    if rgb.ndim != 3 or rgb.shape[0] != 3 or u_map.ndim != 3 or u_map.shape[0] != 1:
        raise AssertionError("Paired geometry expects RGB=[3,H,W], U-map=[1,H,W]")
    if tuple(rgb.shape[-2:]) != tuple(u_map.shape[-2:]):
        raise AssertionError("Paired RGB/U-map geometry requires matching spatial shapes")
    if crop is not None:
        top, left, height, width = (int(value) for value in crop)
        source_h, source_w = rgb.shape[-2:]
        if min(top, left, height, width) < 0 or height == 0 or width == 0:
            raise ValueError(f"Invalid paired crop: {crop}")
        if top + height > source_h or left + width > source_w:
            raise ValueError(f"Paired crop is outside source shape: crop={crop}, shape={(source_h, source_w)}")
        rgb = rgb[:, top : top + height, left : left + width]
        u_map = u_map[:, top : top + height, left : left + width]
    if hflip:
        rgb = torch.flip(rgb, dims=(-1,))
        u_map = torch.flip(u_map, dims=(-1,))
    if resize is not None:
        rgb = _resize(rgb, resize, "bilinear")
        u_map = _resize(u_map, resize, "bilinear")
    return rgb.contiguous(), u_map.contiguous()


class ARQData(Dataset):
    """Load aligned raw, enhanced, uncertainty, mask, and four-task labels."""

    def __init__(
        self,
        csv_path: str,
        raw_dir: str,
        enh_dir: str,
        umap_dir: str,
        mask_dirs: Sequence[str],
        *,
        training: bool = False,
        hflip_prob: Optional[float] = None,
        risk_size: Tuple[int, int] = RISK_SIZE,
        retfound_size: Tuple[int, int] = RETFOUND_SIZE,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        retfound_only: bool = False,
        layer_screen_only: bool = False,
        umap_residual_only: bool = False,
        umap_output_size: Tuple[int, int] = RISK_SIZE,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path).expanduser().resolve()
        self.raw_dir = Path(raw_dir).expanduser().resolve()
        self.enh_dir = Path(enh_dir).expanduser().resolve()
        self.umap_dir = Path(umap_dir).expanduser().resolve()
        self.mask_dirs = [
            Path(directory).expanduser().resolve() for directory in mask_dirs
        ]
        self.retfound_only = bool(retfound_only)
        self.layer_screen_only = bool(layer_screen_only)
        self.umap_residual_only = bool(umap_residual_only)
        self.umap_output_size = tuple(int(value) for value in umap_output_size)
        if self.umap_output_size not in (RISK_SIZE, RETFOUND_SIZE):
            raise ValueError(
                f"U-map output size must be {RISK_SIZE} or {RETFOUND_SIZE}, "
                f"got {self.umap_output_size}"
            )
        if sum((self.retfound_only, self.layer_screen_only, self.umap_residual_only)) > 1:
            raise ValueError(
                "retfound_only, layer_screen_only and umap_residual_only are mutually exclusive"
            )

        if tuple(risk_size) != RISK_SIZE:
            raise ValueError(
                f"ARQ risk_size must remain {RISK_SIZE}, got {tuple(risk_size)}"
            )
        if tuple(retfound_size) != RETFOUND_SIZE:
            raise ValueError(
                "RETFound input size must remain "
                f"{RETFOUND_SIZE}, got {tuple(retfound_size)}"
            )
        if hflip_prob is None:
            hflip_prob = 0.5 if training else 0.0
        if not 0.0 <= hflip_prob <= 1.0:
            raise ValueError(f"hflip_prob must be in [0,1], got {hflip_prob}")
        if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
            raise ValueError(f"Invalid normalization mean/std: mean={mean}, std={std}")

        self.training = bool(training)
        self.hflip_prob = float(hflip_prob)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Required CSV does not exist: {self.csv_path}")
        self.raw_index = _build_stem_index(
            [self.raw_dir], IMAGE_EXTENSIONS, "raw image"
        )
        if self.retfound_only or self.layer_screen_only:
            # The block-screen pipeline is deliberately raw-image-only.  Keep
            # the formal ARQ parsing/preprocessing implementation while never
            # indexing or loading enhanced, uncertainty, or mask inputs.
            self.enh_index = {}
            self.umap_index = {}
            self.mask_index = {}
        elif self.umap_residual_only:
            self.enh_index = {}
            self.umap_index = _build_stem_index(
                [self.umap_dir], (".npy",), "U-map residual uncertainty_raw"
            )
            self.mask_index = {}
        else:
            self.enh_index = _build_stem_index(
                [self.enh_dir], IMAGE_EXTENSIONS, "enhanced image"
            )
            self.umap_index = _build_stem_index(
                [self.umap_dir], (".npy",), "uncertainty_raw"
            )
            self.mask_index = _build_stem_index(
                self.mask_dirs, IMAGE_EXTENSIONS, "fundus mask"
            )

        self.rows = self._read_rows()
        self.samples = [self._resolve_sample(row) for row in self.rows]

    def _read_rows(self) -> List[Mapping[str, str]]:
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            required = ["file_name", *LABEL_COLUMNS.values()]
            if self.retfound_only or self.layer_screen_only or self.umap_residual_only:
                required.append("patient_id")
            missing = [column for column in required if column not in columns]
            if missing:
                raise ValueError(
                    f"CSV {self.csv_path} is missing columns {missing}; "
                    f"actual columns={columns}"
                )
            rows = list(reader)
        if not rows:
            raise ValueError(f"CSV contains no samples: {self.csv_path}")

        seen: Dict[str, str] = {}
        for row in rows:
            file_name = str(row["file_name"]).strip()
            stem = Path(file_name).stem
            if not stem:
                raise ValueError(f"Empty file_name in CSV: {self.csv_path}")
            if stem in seen:
                raise ValueError(
                    f"Duplicate CSV stem {stem!r}: {seen[stem]!r} and {file_name!r}"
                )
            seen[stem] = file_name
        return rows

    def _resolve_sample(self, row: Mapping[str, str]) -> Dict[str, object]:
        file_name = str(row["file_name"]).strip()
        stem = Path(file_name).stem
        paths = {
            "raw": self.raw_index.get(stem),
            "enh": self.enh_index.get(stem),
            "umap": self.umap_index.get(stem),
            "mask": self.mask_index.get(stem),
        }
        if self.retfound_only or self.layer_screen_only:
            required_paths = ("raw",)
        elif self.umap_residual_only:
            required_paths = ("raw", "umap")
        else:
            required_paths = tuple(paths)
        missing = [label for label in required_paths if paths[label] is None]
        if missing:
            raise FileNotFoundError(
                f"Missing ARQ input(s) {missing} for sample name={stem!r}, "
                f"CSV file_name={file_name!r}, csv_path={str(self.csv_path)!r}"
            )

        gt, gt_mask = [], []
        for task in TASK_NAMES:
            column = LABEL_COLUMNS[task]
            text = str(row.get(column, "")).strip()
            try:
                value = float(text)
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                gt.append(0.0)
                gt_mask.append(0.0)
            else:
                if not 0.0 <= value <= 5.0:
                    raise ValueError(
                        f"{task} label outside [0,5] for sample name={stem!r}, "
                        f"path={str(paths['raw'])!r}: actual={value}"
                    )
                gt.append(value)
                gt_mask.append(1.0)

        patient_id = str(row.get("patient_id", "")).strip()
        if (
            self.retfound_only or self.layer_screen_only or self.umap_residual_only
        ) and not patient_id:
            raise ValueError(
                f"Empty patient_id for sample name={stem!r}, "
                f"csv_path={str(self.csv_path)!r}"
            )

        return {
            "name": stem,
            "file_name": file_name,
            "patient_id": patient_id,
            "raw_path": paths["raw"],
            "enh_path": paths["enh"],
            "umap_path": paths["umap"],
            "mask_path": paths["mask"],
            "gt": gt,
            "gt_mask": gt_mask,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        name = str(sample["name"])
        raw_path = Path(sample["raw_path"])
        raw = _resize(_read_rgb(raw_path), RISK_SIZE, "bilinear")

        if self.umap_residual_only:
            umap_path = Path(sample["umap_path"])
            u_map = _resize(_read_umap(umap_path, name), RISK_SIZE, "bilinear")
            do_hflip = self.training and torch.rand(()) < self.hflip_prob
            raw, u_map = apply_paired_geometry(
                raw, u_map, hflip=bool(do_hflip)
            )
            img = _resize(raw, RETFOUND_SIZE, "bilinear")
            if self.umap_output_size != RISK_SIZE:
                u_map = _resize(u_map, self.umap_output_size, "bilinear")
            img = (img - self.mean) / self.std
            gt = torch.tensor(sample["gt"], dtype=torch.float32)
            gt_mask = torch.tensor(sample["gt_mask"], dtype=torch.float32)
            raw_path_text = str(raw_path)
            _assert_shape(raw, (3, 512, 512), "raw", name, raw_path_text)
            _assert_shape(img, (3, 384, 384), "img", name, raw_path_text)
            _assert_shape(
                u_map, (1, *self.umap_output_size), "u_map", name, str(umap_path)
            )
            _assert_shape(gt, (len(TASK_NAMES),), "gt", name, raw_path_text)
            _assert_shape(gt_mask, (len(TASK_NAMES),), "gt_mask", name, raw_path_text)
            for label, tensor in (("raw", raw), ("img", img), ("u_map", u_map)):
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"{label} contains NaN or Inf for sample name={name!r}, "
                        f"path={raw_path_text!r}"
                    )
            # Formal V1 uncertainty_raw is already a globally meaningful,
            # non-negative float32 field.  Do not perform per-image min/max.
            if u_map.min().item() < 0.0:
                raise ValueError(f"u_map contains negative uncertainty: {umap_path}")
            return {
                "img": img.contiguous(),
                "raw": raw.contiguous(),
                "u_map": u_map.contiguous(),
                "gt": gt,
                "gt_mask": gt_mask,
                "name": name,
                "patient_id": str(sample["patient_id"]),
                "path": raw_path_text,
                "u_map_path": str(umap_path),
            }

        if self.retfound_only or self.layer_screen_only:
            do_hflip = self.training and torch.rand(()) < self.hflip_prob
            if bool(do_hflip):
                raw = torch.flip(raw, dims=(-1,))
            img = _resize(raw, RETFOUND_SIZE, "bilinear")
            img = (img - self.mean) / self.std
            gt = torch.tensor(sample["gt"], dtype=torch.float32)
            gt_mask = torch.tensor(sample["gt_mask"], dtype=torch.float32)
            raw_path_text = str(raw_path)
            _assert_shape(img, (3, 384, 384), "img", name, raw_path_text)
            task_count = len(TASK_NAMES)
            _assert_shape(gt, (task_count,), "gt", name, raw_path_text)
            _assert_shape(gt_mask, (task_count,), "gt_mask", name, raw_path_text)
            if not torch.isfinite(img).all():
                raise ValueError(
                    f"img contains NaN or Inf for sample name={name!r}, "
                    f"path={raw_path_text!r}"
                )
            result = {
                "img": img.contiguous(),
                "gt": gt,
                "gt_mask": gt_mask,
                "name": name,
                "patient_id": str(sample["patient_id"]),
                "path": raw_path_text,
            }
            if self.layer_screen_only:
                _assert_shape(raw, (3, 512, 512), "raw", name, raw_path_text)
                if not torch.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
                    raise ValueError(
                        f"raw outside [0,1] or non-finite for sample name={name!r}, "
                        f"path={raw_path_text!r}"
                    )
                result["raw"] = raw.contiguous()
            return result

        enh_path = Path(sample["enh_path"])
        umap_path = Path(sample["umap_path"])
        mask_path = Path(sample["mask_path"])
        enh = _resize(_read_rgb(enh_path), RISK_SIZE, "bilinear")
        umap = _resize(_read_umap(umap_path, name), RISK_SIZE, "bilinear")
        mask = _resize(_read_mask(mask_path), RISK_SIZE, "nearest")
        mask = (mask > 0.5).to(dtype=torch.float32)

        do_hflip = self.training and torch.rand(()) < self.hflip_prob
        if bool(do_hflip):
            raw = torch.flip(raw, dims=(-1,))
            enh = torch.flip(enh, dims=(-1,))
            umap = torch.flip(umap, dims=(-1,))
            mask = torch.flip(mask, dims=(-1,))

        img = _resize(raw, RETFOUND_SIZE, "bilinear")
        img = (img - self.mean) / self.std
        gt = torch.tensor(sample["gt"], dtype=torch.float32)
        gt_mask = torch.tensor(sample["gt_mask"], dtype=torch.float32)

        raw_path_text = str(raw_path)
        _assert_shape(raw, (3, 512, 512), "raw", name, raw_path_text)
        _assert_shape(enh, (3, 512, 512), "enh", name, str(enh_path))
        _assert_shape(umap, (1, 512, 512), "umap", name, str(umap_path))
        _assert_shape(mask, (1, 512, 512), "mask", name, str(mask_path))
        _assert_shape(img, (3, 384, 384), "img", name, raw_path_text)
        task_count = len(TASK_NAMES)
        _assert_shape(gt, (task_count,), "gt", name, raw_path_text)
        _assert_shape(gt_mask, (task_count,), "gt_mask", name, raw_path_text)
        if enh.shape != raw.shape:
            raise _shape_error("enh versus raw", enh.shape, raw.shape, name, str(enh_path))
        if mask.shape != umap.shape:
            raise _shape_error(
                "mask versus umap", mask.shape, umap.shape, name, str(mask_path)
            )
        if not torch.isfinite(raw).all() or raw.min() < 0 or raw.max() > 1:
            raise ValueError(
                f"raw outside [0,1] or non-finite for sample name={name!r}, "
                f"path={raw_path_text!r}"
            )
        if not torch.isfinite(enh).all() or enh.min() < 0 or enh.max() > 1:
            raise ValueError(
                f"enh outside [0,1] or non-finite for sample name={name!r}, "
                f"path={str(enh_path)!r}"
            )

        return {
            "img": img.contiguous(),
            "raw": raw.contiguous(),
            "enh": enh.contiguous(),
            "umap": umap.contiguous(),
            "mask": mask.contiguous(),
            "gt": gt,
            "gt_mask": gt_mask,
            "name": name,
            "path": raw_path_text,
        }

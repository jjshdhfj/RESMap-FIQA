from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from arq.constants import TASK_NAMES
from arq.data import (
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_COLUMNS,
    RETFOUND_SIZE,
    RISK_SIZE,
    _build_stem_index,
    _read_rgb,
    _resize,
)
from arq.a3_geometry import apply_joint_geometry


U_EXTENSIONS = (".png", ".tif", ".tiff")


def _read_csv(path: Path):
    last_error = None

    for enc in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ):
        try:
            with path.open(
                "r",
                encoding=enc,
                newline="",
            ) as f:
                return list(csv.DictReader(f))

        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Cannot decode CSV: {path}"
    ) from last_error


def read_umap_u16(
    path: Path,
    name: str,
):
    """
    Formal A3 U-map decoding.

        uint16 -> float32 / 65535

    No per-image normalization.
    No min-max.
    No log.
    No percentile clipping.

    The source U-map has already been masked outside FOV
    during uncertainty generation.
    """

    arr = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if arr is None:
        raise RuntimeError(
            f"{name}: failed to read U-map "
            f"{path}"
        )

    if arr.dtype != np.uint16:
        raise RuntimeError(
            f"{name}: expected uint16, "
            f"got {arr.dtype}"
        )

    if arr.ndim == 3:
        if arr.shape[2] != 1:
            raise RuntimeError(
                f"{name}: expected single-channel U-map, "
                f"got {arr.shape}"
            )
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise RuntimeError(
            f"{name}: unexpected U-map shape "
            f"{arr.shape}"
        )

    u = (
        arr.astype(np.float32)
        / 65535.0
    )

    if not np.isfinite(u).all():
        raise RuntimeError(
            f"{name}: non-finite U values"
        )

    if (
        float(u.min()) < 0.0
        or
        float(u.max()) > 1.0
    ):
        raise RuntimeError(
            f"{name}: U-map outside [0,1]"
        )

    return (
        torch.from_numpy(u)
        .unsqueeze(0)
        .float()
    )


class Final72RGBUMapDataset(Dataset):
    """
    A3 formal dataset.

    RGB processing is kept aligned with the earlier
    ProgressiveEvidenceData pipeline.

    GT remains ORIGINAL 0~5.
    Trainer performs /5 exactly once.
    """

    def __init__(
        self,
        csv_path,
        raw_dir,
        umap_dir,
        *,
        training=False,
        allow_test=False,
    ):
        super().__init__()

        self.csv_path = Path(
            csv_path
        ).expanduser().resolve()

        self.raw_dir = Path(
            raw_dir
        ).expanduser().resolve()

        self.umap_dir = Path(
            umap_dir
        ).expanduser().resolve()

        self.training = bool(
            training
        )

        if (
            "test"
            in self.csv_path.name.lower()
            and not allow_test
        ):
            raise PermissionError(
                "Formal A3 training/validation "
                "cannot access test CSV."
            )

        if not self.csv_path.is_file():
            raise FileNotFoundError(
                self.csv_path
            )

        if not self.raw_dir.is_dir():
            raise FileNotFoundError(
                self.raw_dir
            )

        if not self.umap_dir.is_dir():
            raise FileNotFoundError(
                self.umap_dir
            )

        self.raw_index = _build_stem_index(
            [self.raw_dir],
            IMAGE_EXTENSIONS,
            "raw image",
        )

        self.umap_index = _build_stem_index(
            [self.umap_dir],
            U_EXTENSIONS,
            "uncertainty_scale_u16",
        )

        rows = _read_csv(
            self.csv_path
        )

        if not rows:
            raise RuntimeError(
                f"Empty CSV: {self.csv_path}"
            )

        if "file_name" not in rows[0]:
            raise KeyError(
                "Runtime CSV must contain file_name. "
                f"Columns={list(rows[0].keys())}"
            )

        self.samples = []
        seen = set()

        for row in rows:

            name = Path(
                str(row["file_name"])
            ).stem

            if name in seen:
                raise RuntimeError(
                    f"Duplicate stem in CSV: {name}"
                )

            seen.add(name)

            raw_path = self.raw_index.get(
                name
            )

            u_path = self.umap_index.get(
                name
            )

            if raw_path is None:
                raise FileNotFoundError(
                    f"Missing RGB: {name}"
                )

            if u_path is None:
                raise FileNotFoundError(
                    f"Missing U-map: {name}"
                )

            gt = []
            gt_mask = []

            for task in TASK_NAMES:

                column = LABEL_COLUMNS[
                    task
                ]

                try:
                    value = float(
                        row[column]
                    )
                except (
                    KeyError,
                    ValueError,
                    TypeError,
                ):
                    value = float("nan")

                if math.isfinite(value):
                    # IMPORTANT:
                    # ORIGINAL 0~5.
                    # Do NOT divide here.
                    gt.append(value)
                    gt_mask.append(1.0)
                else:
                    gt.append(0.0)
                    gt_mask.append(0.0)

            self.samples.append(
                {
                    "name": name,
                    "patient_id": str(
                        row.get(
                            "patient_id",
                            "",
                        )
                    ),
                    "raw_path": raw_path,
                    "u_path": u_path,
                    "gt": gt,
                    "gt_mask": gt_mask,
                }
            )

        self.mean = torch.tensor(
            IMAGENET_MEAN,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

        self.std = torch.tensor(
            IMAGENET_STD,
            dtype=torch.float32,
        ).view(
            3,
            1,
            1,
        )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        s = self.samples[index]

        # ----------------------------------------------------
        # RGB: same common 512 frame
        # ----------------------------------------------------
        raw = _resize(
            _read_rgb(
                s["raw_path"]
            ),
            RISK_SIZE,
            "bilinear",
        )

        # ----------------------------------------------------
        # U: globally encoded uint16 -> [0,1]
        # ----------------------------------------------------
        u = read_umap_u16(
            s["u_path"],
            s["name"],
        )

        if (
            u.shape[-2:]
            != RISK_SIZE
        ):
            u = _resize(
                u,
                RISK_SIZE,
                "bilinear",
            )

        # ----------------------------------------------------
        # Joint geometry:
        # one random hflip decision shared by RGB and U.
        # ----------------------------------------------------
        dummy1 = torch.zeros(
            1,
            *RISK_SIZE,
            dtype=torch.float32,
        )

        dummy2 = torch.ones(
            1,
            *RISK_SIZE,
            dtype=torch.float32,
        )

        do_hflip = (
            self.training
            and bool(
                torch.rand(()) < 0.5
            )
        )

        raw, u, _, _ = (
            apply_joint_geometry(
                raw,
                u,
                dummy1,
                dummy2,
                hflip=do_hflip,
            )
        )

        # ----------------------------------------------------
        # 384 input
        # ----------------------------------------------------
        rgb01 = _resize(
            raw,
            RETFOUND_SIZE,
            "bilinear",
        )

        u = _resize(
            u,
            RETFOUND_SIZE,
            "bilinear",
        )

        img = (
            rgb01
            - self.mean
        ) / self.std

        if tuple(img.shape) != (
            3,
            384,
            384,
        ):
            raise RuntimeError(
                f"{s['name']}: RGB shape "
                f"{tuple(img.shape)}"
            )

        if tuple(u.shape) != (
            1,
            384,
            384,
        ):
            raise RuntimeError(
                f"{s['name']}: U shape "
                f"{tuple(u.shape)}"
            )

        return {
            "img":
                img.contiguous(),

            "u_map":
                u.float().contiguous(),

            # ORIGINAL 0~5
            "gt":
                torch.tensor(
                    s["gt"],
                    dtype=torch.float32,
                ),

            "gt_mask":
                torch.tensor(
                    s["gt_mask"],
                    dtype=torch.float32,
                ),

            "name":
                s["name"],

            "patient_id":
                s["patient_id"],

            "path":
                str(
                    s["raw_path"]
                ),
        }

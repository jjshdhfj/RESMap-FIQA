#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official A3-only locked-test evaluator.

The dataset, preprocessing, EMA checkpoint loading, 0..5 scaling,
and metrics are taken directly from the single-seed formal A0--A3
source. Only non-A3 model branches and multi-model aggregation are removed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import Dataset, DataLoader

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
from arq.retfound_b15_taskquery_umap_4head import RETFoundB15TaskQueryUMap4Head

# Original locked-test defaults, retained for direct numerical comparison.
ORIGINAL_ROOT = Path("/home/ly/jbk/RETFound-main")
SPLIT_ROOT = ORIGINAL_ROOT / "runs/locked_patient_split_seed20260821/final_patient_72_8_20"
RUN_DIR = ORIGINAL_ROOT / "runs/final_patient_72_8_20/seed3407/a3_b15_taskquery_umap"
RGB_DIR = Path("/home/ly/jbk/data/tiqu_matched_1024x1024_from_original_csv")
U_DIR = Path("/home/ly/jbk/data/gfenet_uhead_v1_real_512_cpu/uncertainty_scale_u16")
RET_CKPT = ORIGINAL_ROOT / "RETFound_mae_natureCFP.pth"
OUT_ROOT = RUN_DIR / "locked_test_a3_only"
RUN_NAME = "A3_B15_TaskQuery_UMap"
STEPS_PER_EPOCH = 1103
EXPECTED_TEST = 1227
TASKS = list(TASK_NAMES)

def read_csv_auto(path):
    errors = []

    for enc in (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "gbk",
    ):
        try:
            df = pd.read_csv(
                path,
                encoding=enc,
            )

            print(
                f"[CSV] {path} "
                f"encoding={enc} "
                f"rows={len(df)}"
            )

            return df

        except Exception as exc:
            errors.append(
                f"{enc}: {exc}"
            )

    raise RuntimeError(
        f"Cannot read {path}\n"
        + "\n".join(errors)
    )



def resolve_test_csv():

    preferred = [
        SPLIT_ROOT / "test_runtime.csv",
        SPLIT_ROOT / "test.csv",
        SPLIT_ROOT / "locked_test.csv",
        SPLIT_ROOT / "test_split.csv",
    ]

    for p in preferred:

        if not p.is_file():
            continue

        df = read_csv_auto(p)

        if len(df) == EXPECTED_TEST:
            print(
                "[TEST CSV]",
                p,
            )
            return p

    candidates = sorted(
        SPLIT_ROOT.glob("*.csv")
    )

    for p in candidates:

        try:
            df = read_csv_auto(p)
        except Exception:
            continue

        if len(df) == EXPECTED_TEST:

            print(
                "[TEST CSV AUTO]",
                p,
            )

            return p

    raise RuntimeError(
        "Cannot locate the formal "
        "1227-image locked test CSV.\n"
        f"Directory: {SPLIT_ROOT}"
    )



def resolve_file_column(df):

    candidates = (
        "file_name",
        "图片文件名",
        "image_name",
        "filename",
    )

    for c in candidates:

        if c in df.columns:
            return c

    raise KeyError(
        "Cannot resolve filename column.\n"
        f"Columns={df.columns.tolist()}"
    )



def resolve_label_column(
    df,
    task,
):

    candidates = []

    if task in LABEL_COLUMNS:
        candidates.append(
            LABEL_COLUMNS[task]
        )

    candidates.extend(
        [
            task,
            task.lower(),
            task.capitalize(),
            f"gt_{task}",
            f"{task}_gt",
        ]
    )

    for c in candidates:

        if c in df.columns:
            return c

    raise KeyError(
        f"Cannot resolve label column "
        f"for task={task}\n"
        f"Candidates={candidates}\n"
        f"Columns={df.columns.tolist()}"
    )



class LockedTestDataset(Dataset):

    def __init__(
        self,
        csv_path,
        raw_dir,
        umap_dir,
    ):
        super().__init__()

        self.df = read_csv_auto(
            csv_path
        )

        if len(self.df) != EXPECTED_TEST:
            raise RuntimeError(
                f"Expected {EXPECTED_TEST} test images, "
                f"got {len(self.df)}"
            )

        self.file_col = (
            resolve_file_column(
                self.df
            )
        )

        self.label_cols = {
            task:
                resolve_label_column(
                    self.df,
                    task,
                )
            for task in TASKS
        }

        print(
            "[FILE COLUMN]",
            self.file_col,
        )

        print(
            "[LABEL COLUMNS]",
            self.label_cols,
        )

        self.raw_index = (
            _build_stem_index(
                [Path(raw_dir)],
                IMAGE_EXTENSIONS,
                "raw image",
            )
        )

        self.u_index = (
            _build_stem_index(
                [Path(umap_dir)],
                (
                    ".png",
                    ".tif",
                    ".tiff",
                ),
                "U-map",
            )
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

        self.samples = []

        missing_rgb = []
        missing_u = []

        for _, row in (
            self.df.iterrows()
        ):

            file_name = str(
                row[self.file_col]
            ).strip()

            stem = Path(
                file_name
            ).stem

            raw_path = (
                self.raw_index
                .get(stem)
            )

            u_path = (
                self.u_index
                .get(stem)
            )

            if raw_path is None:
                missing_rgb.append(
                    file_name
                )

            if u_path is None:
                missing_u.append(
                    file_name
                )

            gt = []
            mask = []

            for task in TASKS:

                v = pd.to_numeric(
                    pd.Series(
                        [
                            row[
                                self.label_cols[
                                    task
                                ]
                            ]
                        ]
                    ),
                    errors="coerce",
                ).iloc[0]

                if pd.isna(v):
                    gt.append(0.0)
                    mask.append(0.0)
                else:
                    gt.append(
                        float(v)
                    )
                    mask.append(1.0)

            self.samples.append(
                {
                    "file_name":
                        file_name,

                    "stem":
                        stem,

                    "raw_path":
                        raw_path,

                    "u_path":
                        u_path,

                    "gt":
                        gt,

                    "mask":
                        mask,
                }
            )

        if missing_rgb:
            raise RuntimeError(
                "Missing RGB examples:\n"
                + "\n".join(
                    missing_rgb[:20]
                )
            )

        if missing_u:
            raise RuntimeError(
                "Missing U-map examples:\n"
                + "\n".join(
                    missing_u[:20]
                )
            )

        print(
            "[MATCH] RGB =",
            len(self.samples),
        )

        print(
            "[MATCH] U   =",
            len(self.samples),
        )

    def __len__(self):
        return len(
            self.samples
        )

    def _read_u(
        self,
        path,
        stem,
    ):

        arr = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if arr is None:
            raise RuntimeError(
                f"Cannot read U-map: {path}"
            )

        if arr.dtype != np.uint16:
            raise RuntimeError(
                f"{stem}: expected uint16 U-map, "
                f"got {arr.dtype}"
            )

        if arr.ndim == 3:

            if arr.shape[2] != 1:
                raise RuntimeError(
                    f"{stem}: invalid U shape "
                    f"{arr.shape}"
                )

            arr = arr[..., 0]

        u = (
            arr.astype(np.float32)
            / 65535.0
        )

        if not np.isfinite(u).all():
            raise RuntimeError(
                f"{stem}: non-finite U"
            )

        u = (
            torch.from_numpy(u)
            .unsqueeze(0)
            .float()
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

        u = _resize(
            u,
            RETFOUND_SIZE,
            "bilinear",
        )

        return u.contiguous()

    def __getitem__(
        self,
        idx,
    ):

        s = self.samples[idx]

        # ---------------------------------------------
        # Deterministic RGB preprocessing.
        # NO train augmentation on Locked Test.
        # ---------------------------------------------

        rgb = _read_rgb(
            s["raw_path"]
        )

        rgb = _resize(
            rgb,
            RISK_SIZE,
            "bilinear",
        )

        rgb = _resize(
            rgb,
            RETFOUND_SIZE,
            "bilinear",
        )

        img = (
            rgb
            - self.mean
        ) / self.std

        u = self._read_u(
            s["u_path"],
            s["stem"],
        )

        return {
            "img":
                img.contiguous(),

            "u_map":
                u,

            "gt":
                torch.tensor(
                    s["gt"],
                    dtype=torch.float32,
                ),

            "gt_mask":
                torch.tensor(
                    s["mask"],
                    dtype=torch.float32,
                ),

            "file_name":
                s["file_name"],
        }



def safe_srcc(
    gt,
    pred,
):

    if (
        len(gt) < 2
        or
        np.std(gt) == 0
        or
        np.std(pred) == 0
    ):
        return float("nan")

    return float(
        spearmanr(
            gt,
            pred,
        )[0]
    )



def safe_plcc(
    gt,
    pred,
):

    if (
        len(gt) < 2
        or
        np.std(gt) == 0
        or
        np.std(pred) == 0
    ):
        return float("nan")

    return float(
        pearsonr(
            gt,
            pred,
        )[0]
    )



def compute_metrics(
    gt,
    pred,
    mask,
):

    metrics = {}

    srccs = []
    plccs = []
    rmses = []
    maes = []

    for i, task in enumerate(
        TASKS
    ):

        valid = (
            mask[:, i] > 0.5
        )

        g = gt[
            valid,
            i,
        ]

        p = pred[
            valid,
            i,
        ]

        srcc = safe_srcc(
            g,
            p,
        )

        plcc = safe_plcc(
            g,
            p,
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (p - g) ** 2
                )
            )
        )

        mae = float(
            np.mean(
                np.abs(
                    p - g
                )
            )
        )

        count = int(
            valid.sum()
        )

        metrics[
            f"{task}_SRCC"
        ] = srcc

        metrics[
            f"{task}_PLCC"
        ] = plcc

        metrics[
            f"{task}_RMSE"
        ] = rmse

        metrics[
            f"{task}_MAE"
        ] = mae

        metrics[
            f"{task}_count"
        ] = count

        srccs.append(srcc)
        plccs.append(plcc)
        rmses.append(rmse)
        maes.append(mae)

    metrics["mean_SRCC"] = float(
        np.mean(srccs)
    )

    metrics["mean_PLCC"] = float(
        np.mean(plccs)
    )

    metrics["mean_RMSE"] = float(
        np.mean(rmses)
    )

    metrics["mean_MAE"] = float(
        np.mean(maes)
    )

    # CEA = Clarity + Exposure + Artifact
    metrics["CEA_mean_SRCC"] = float(
        np.mean(
            srccs[:3]
        )
    )

    metrics["CEA_mean_PLCC"] = float(
        np.mean(
            plccs[:3]
        )
    )

    metrics["N"] = int(
        gt.shape[0]
    )

    return metrics



def load_best_ema(
    model,
    checkpoint_path,
):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            checkpoint_path
        )

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if "ema" not in ckpt:
        raise KeyError(
            f"No EMA in checkpoint: "
            f"{checkpoint_path}"
        )

    if "shadow" not in ckpt["ema"]:
        raise KeyError(
            "EMA has no shadow state"
        )

    shadow = ckpt[
        "ema"
    ]["shadow"]

    named_params = dict(
        model.named_parameters()
    )

    trainable_names = {
        name
        for name, p
        in model.named_parameters()
        if p.requires_grad
    }

    shadow_names = set(
        shadow.keys()
    )

    missing = (
        trainable_names
        - shadow_names
    )

    extra = (
        shadow_names
        - set(
            named_params.keys()
        )
    )

    if missing:
        raise RuntimeError(
            "EMA missing trainable params:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    if extra:
        raise RuntimeError(
            "EMA contains unknown params:\n"
            + "\n".join(
                sorted(extra)
            )
        )

    with torch.no_grad():

        for name, value in (
            shadow.items()
        ):

            named_params[
                name
            ].copy_(
                value.to(
                    dtype=
                        named_params[
                            name
                        ].dtype
                )
            )

    completed_steps = int(
        ckpt.get(
            "completed_steps",
            -1,
        )
    )

    best_epoch = (
        completed_steps
        // STEPS_PER_EPOCH
        if completed_steps > 0
        else
        int(
            ckpt.get(
                "epoch",
                -1,
            )
        ) + 1
    )

    val_metrics = ckpt.get(
        "metrics",
        {}
    )

    val_srccs = [
        val_metrics.get(
            f"{task}_SRCC",
            float("nan"),
        )
        for task in TASKS
    ]

    best_val_mean_srcc = float(
        np.nanmean(
            val_srccs
        )
    )

    return (
        ckpt,
        best_epoch,
        completed_steps,
        best_val_mean_srcc,
    )



def build_model():
    return RETFoundB15TaskQueryUMap4Head(
        checkpoint_path=str(RET_CKPT),
        image_size=384,
    )


@torch.no_grad()
def test_one(loader, device):

    name = RUN_NAME
    run_dir = RUN_DIR

    best_ckpt = (
        run_dir
        / "best_srcc.pt"
    )

    latest_ckpt = (
        run_dir
        / "latest.pt"
    )

    if not best_ckpt.is_file():
        raise FileNotFoundError(
            f"{name}: missing best checkpoint\n"
            f"{best_ckpt}"
        )

    # Formal testing should be done after full training.
    if latest_ckpt.is_file():

        latest = torch.load(
            latest_ckpt,
            map_location="cpu",
        )

        latest_steps = int(
            latest.get(
                "completed_steps",
                -1,
            )
        )

        print(
            f"[{name}] latest completed_steps = "
            f"{latest_steps}"
        )

        if latest_steps != 165450:

            raise RuntimeError(
                f"{name} has not completed the formal "
                f"165450-step training budget.\n"
                f"Current latest steps={latest_steps}\n"
                "Do not evaluate Locked Test yet."
            )

        del latest

    model = build_model().to(device)

    (
        ckpt,
        best_epoch,
        best_step,
        best_val_mean_srcc,
    ) = load_best_ema(
        model,
        best_ckpt,
    )

    model.eval()

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)
    print(
        "best checkpoint =",
        best_ckpt,
    )
    print(
        "best epoch      =",
        best_epoch,
    )
    print(
        "best step       =",
        best_step,
    )
    print(
        "Val Mean SRCC   =",
        f"{best_val_mean_srcc:.6f}",
    )

    preds = []
    gts = []
    masks = []
    names = []

    for batch in loader:

        img = (
            batch["img"]
            .float()
            .to(
                device,
                non_blocking=True,
            )
        )

        u_map = (
            batch["u_map"]
            .float()
            .to(device, non_blocking=True)
        )
        pred01 = model(img, u_map)

        # Restore to original 0~5 scale.
        pred5 = (
            pred01
            .detach()
            .float()
            .cpu()
            .numpy()
            * 5.0
        )

        preds.append(
            pred5
        )

        gts.append(
            batch["gt"]
            .float()
            .numpy()
        )

        masks.append(
            batch["gt_mask"]
            .float()
            .numpy()
        )

        names.extend(
            batch[
                "file_name"
            ]
        )

    pred = np.concatenate(
        preds,
        axis=0,
    )

    gt = np.concatenate(
        gts,
        axis=0,
    )

    mask = np.concatenate(
        masks,
        axis=0,
    )

    if len(pred) != EXPECTED_TEST:
        raise RuntimeError(
            f"{name}: expected "
            f"{EXPECTED_TEST} predictions, "
            f"got {len(pred)}"
        )

    metrics = compute_metrics(
        gt,
        pred,
        mask,
    )

    metrics[
        "best_epoch"
    ] = int(
        best_epoch
    )

    metrics[
        "best_step"
    ] = int(
        best_step
    )

    metrics[
        "val_mean_SRCC_at_best"
    ] = (
        best_val_mean_srcc
    )

    # --------------------------------------------------------
    # Save per-image predictions
    # --------------------------------------------------------

    records = []

    for n in range(
        len(names)
    ):

        row = {
            "file_name":
                names[n]
        }

        for i, task in enumerate(
            TASKS
        ):

            row[
                f"gt_{task}"
            ] = float(
                gt[n, i]
            )

            row[
                f"pred_{task}"
            ] = float(
                pred[n, i]
            )

            row[
                f"valid_{task}"
            ] = int(
                mask[n, i]
                > 0.5
            )

        records.append(
            row
        )

    pd.DataFrame(
        records
    ).to_csv(
        OUT_ROOT
        / f"{name}_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        OUT_ROOT
        / f"{name}_test_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    del model
    del ckpt

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics



def main():
    global SPLIT_ROOT, RUN_DIR, RGB_DIR, U_DIR, RET_CKPT, OUT_ROOT
    ap = argparse.ArgumentParser(description="Formal A3-only locked-test evaluation")
    ap.add_argument("--split-root", type=Path, default=SPLIT_ROOT)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--rgb-dir", type=Path, default=RGB_DIR)
    ap.add_argument("--umap-dir", type=Path, default=U_DIR)
    ap.add_argument("--retfound-ckpt", type=Path, default=RET_CKPT)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <run-dir>/locked_test_a3_only")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--preflight", action="store_true",
                    help="Check files, image mapping, and 1227-case test protocol; no model inference")
    args = ap.parse_args()

    SPLIT_ROOT = args.split_root.expanduser().resolve()
    RUN_DIR = args.run_dir.expanduser().resolve()
    RGB_DIR = args.rgb_dir.expanduser().resolve()
    U_DIR = args.umap_dir.expanduser().resolve()
    RET_CKPT = args.retfound_ckpt.expanduser().resolve()
    OUT_ROOT = (args.out_dir.expanduser().resolve() if args.out_dir
                else RUN_DIR / "locked_test_a3_only")
    if args.batch_size < 1 or args.num_workers < 0:
        ap.error("batch-size must be >=1 and num-workers must be >=0")

    test_csv = resolve_test_csv()
    dataset = LockedTestDataset(test_csv, RGB_DIR, U_DIR)
    if len(dataset) != EXPECTED_TEST:
        raise RuntimeError(f"Expected {EXPECTED_TEST}, got {len(dataset)}")
    best_ckpt = RUN_DIR / "best_srcc.pt"
    for p in [RET_CKPT, best_ckpt]:
        if not p.is_file():
            raise FileNotFoundError(p)
    print("[A3] test_csv=", test_csv)
    print("[A3] run_dir=", RUN_DIR)
    print("[A3] retfound_ckpt=", RET_CKPT)
    print("[A3] expected_test=", EXPECTED_TEST)
    if args.preflight:
        # Verify one actual sample is decoded by the unchanged locked-test dataset.
        sample = dataset[0]
        assert tuple(sample["img"].shape) == (3, 384, 384)
        assert tuple(sample["u_map"].shape) == (1, 384, 384)
        print("[PREFLIGHT] PASS: A3-only files, dataset, first image, U-map and GT")
        return

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    device = torch.device(args.device)
    print("[A3 ONLY LOCKED TEST]", "device=", device, "N=", len(dataset))
    metrics = test_one(loader, device)
    summary = {"Experiment": RUN_NAME,
               "BestEpoch": metrics["best_epoch"],
               "ValMeanSRCC": metrics["val_mean_SRCC_at_best"]}
    for task in TASKS:
        for metric in ("SRCC", "PLCC", "RMSE", "MAE"):
            summary[f"{task}_{metric}"] = metrics[f"{task}_{metric}"]
    for key in ("mean_SRCC", "mean_PLCC", "mean_RMSE", "mean_MAE",
                "CEA_mean_SRCC", "CEA_mean_PLCC", "N"):
        summary[key] = metrics[key]
    pd.DataFrame([summary]).to_csv(
        OUT_ROOT / "a3_locked_test_metrics.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("[SAVED]", OUT_ROOT)

if __name__ == "__main__":
    main()

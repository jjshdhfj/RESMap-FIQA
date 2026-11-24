#!/usr/bin/env python3

import argparse
import os
import contextlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import yaml
from scipy.stats import spearmanr, pearsonr

# 只复用现有数据读取，保证 Split1、resize、normalize 等完全一致

# 纯 baseline 模型：
# Frozen RETFound -> native GAP -> 1024-D -> 4 x Linear(1024,1)
from arq.retfound_b15_taskquery_umap_4head import RETFoundB15TaskQueryUMap4Head
from arq.final72_rgb_umap_dataset import Final72RGBUMapDataset
from arq.a3_sampler import (
    EpochShuffleSampler,
    seed_worker,
)


TASKS = (
    "clarity",
    "exposure",
    "artifact",
    "completeness",
)

SEED = int(os.environ.get("RUN_SEED", "3407"))
TOTAL_ITERS = 165450
VAL_FREQ = 1103
PRINT_FREQ = 50

BASE_LR = 1e-4
MILESTONES = ()
GAMMA = 0.5

EMA_DECAY = 0.99
SMOOTH_L1_BETA = 0.1

# ============================================================
# Seed
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 简单 EMA
# 只维护 requires_grad=True 的参数。
# 本 baseline 中就是四个 Linear heads。
# ============================================================

class SimpleEMA:

    def __init__(self, model, decay=0.99):
        self.decay = float(decay)

        self.shadow = {
            name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model):

        for name, p in model.named_parameters():

            if not p.requires_grad:
                continue

            self.shadow[name].mul_(
                self.decay
            ).add_(
                p.detach(),
                alpha=1.0 - self.decay,
            )

    @contextlib.contextmanager
    def apply_to(self, model):

        backup = {}

        with torch.no_grad():

            for name, p in model.named_parameters():

                if not p.requires_grad:
                    continue

                backup[name] = p.detach().clone()

                p.copy_(
                    self.shadow[name]
                )

        try:
            yield

        finally:

            with torch.no_grad():

                for name, p in model.named_parameters():

                    if name in backup:
                        p.copy_(
                            backup[name]
                        )

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {
                k: v.detach().cpu()
                for k, v in self.shadow.items()
            },
        }


# ============================================================
# Learning rate
#
# Final72 controlled ablation protocol:
#
# A0 / A1 / A2 / A3 all use the same fixed learning rate:
#
#     LR = 1e-4
#
# MILESTONES is empty, so:
#
#     drops = 0
#     lr = BASE_LR * GAMMA ** 0
#        = BASE_LR
#        = 1e-4
#
# set_optimizer_lr() is still called before every optimizer
# update for compatibility with the common trainer structure,
# but the actual learning rate remains unchanged throughout
# all 165450 training iterations.
# ============================================================

def lr_for_step(completed_steps):

    drops = sum(
        completed_steps >= m
        for m in MILESTONES
    )

    return BASE_LR * (
        GAMMA ** drops
    )


def set_optimizer_lr(
    optimizer,
    completed_steps,
):

    lr = lr_for_step(
        completed_steps
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


# ============================================================
# Loss
#
# GT 输入是原始 0~5。
#
# 1. target = gt / 5
# 2. SmoothL1(beta=0.1)
# 3. 每个任务独立按照 gt_mask 求均值
# 4. 四个任务等权平均
#
# ============================================================

def four_task_loss(
    pred,
    gt,
    gt_mask,
):

    target = gt / 5.0

    element = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=SMOOTH_L1_BETA,
    )

    task_losses = []

    parts = {}

    for i, task in enumerate(TASKS):

        valid = gt_mask[:, i]

        count = valid.sum()

        if count.item() > 0:

            loss_t = (
                element[:, i]
                * valid
            ).sum() / count

        else:

            loss_t = (
                pred[:, i].sum()
                * 0.0
            )

        task_losses.append(
            loss_t
        )

        parts[
            f"loss_{task}"
        ] = loss_t

    loss_total = torch.stack(
        task_losses
    ).mean()

    return loss_total, parts


# ============================================================
# Metrics
# ============================================================

def safe_srcc(gt, pred):

    if len(gt) < 2:
        return float("nan")

    if (
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


def safe_plcc(gt, pred):

    if len(gt) < 2:
        return float("nan")

    if (
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


@torch.no_grad()
def validate(
    model,
    loader,
    device,
    ema,
):

    model.eval()

    all_pred = []
    all_gt = []
    all_mask = []

    # validation 使用 EMA heads
    with ema.apply_to(model):

        for batch in loader:

            img = batch[
                "img"
            ].to(
                device,
                non_blocking=True,
            )

            u_map = batch[
                "u_map"
            ].float().to(
                device,
                non_blocking=True,
            )

            gt = (
                batch["gt"]
                .float()
                .cpu()
                .numpy()
            )

            mask = (
                batch["gt_mask"]
                .float()
                .cpu()
                .numpy()
            )

            pred01 = model(
                img,
                u_map,
            )

            # 指标恢复到原始 0~5 分制
            pred5 = (
                pred01
                .detach()
                .float()
                .cpu()
                .numpy()
                * 5.0
            )

            all_pred.append(
                pred5
            )

            all_gt.append(
                gt
            )

            all_mask.append(
                mask
            )

    pred = np.concatenate(
        all_pred,
        axis=0,
    )

    gt = np.concatenate(
        all_gt,
        axis=0,
    )

    mask = np.concatenate(
        all_mask,
        axis=0,
    )

    metrics = {}

    srccs = []
    plccs = []
    rmses = []
    maes = []

    for i, task in enumerate(TASKS):

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
        ] = int(
            valid.sum()
        )

        srccs.append(srcc)
        plccs.append(plcc)
        rmses.append(rmse)
        maes.append(mae)

    metrics[
        "mean_SRCC"
    ] = float(
        np.nanmean(srccs)
    )

    metrics[
        "mean_PLCC"
    ] = float(
        np.nanmean(plccs)
    )

    metrics[
        "mean_RMSE"
    ] = float(
        np.nanmean(rmses)
    )

    metrics[
        "mean_MAE"
    ] = float(
        np.nanmean(maes)
    )

    metrics[
        "CEA_Mean_SRCC"
    ] = float(
        np.nanmean(
            srccs[:3]
        )
    )

    metrics[
        "num_samples"
    ] = int(
        len(gt)
    )

    return metrics


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path,
    model,
    ema,
    optimizer,
    step,
    epoch,
    metrics=None,
):

    torch.save(
        {
            "model":
                model.state_dict(),

            "ema":
                ema.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "completed_steps":
                int(step),

            "epoch":
                int(epoch),

            "metrics":
                metrics,
        },
        path,
    )


# ============================================================
# Resolve RETFound checkpoint
# ============================================================

def resolve_ret_ckpt(cfg):

    paths = cfg.get(
        "paths",
        {},
    )

    candidates = (
        "ret_ckpt",
        "retfound_ckpt",
        "retfound_pretrained",
    )

    for key in candidates:

        value = paths.get(
            key
        )

        if value:
            p = Path(value)

            if p.is_file():
                return str(p)

    # PORTABILITY PATCH 3/3: explicit RETFound path only, no local fallback.
    raise FileNotFoundError(
        "Cannot resolve RETFound checkpoint "
        f"from config paths={paths}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Use V1a-L20K-NoES resolved config only for data paths/preprocessing.",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()


    config_path = Path(
        args.config
    ).expanduser().resolve()

    output_dir = Path(
        args.output
    ).expanduser().resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Refuse accidental overwrite
    # --------------------------------------------------------

    if (
        output_dir
        / "best_srcc.pt"
    ).exists():

        raise RuntimeError(
            "best_srcc.pt already exists:\n"
            f"{output_dir}\n"
            "Refusing accidental overwrite."
        )


    cfg = yaml.safe_load(
        config_path.read_text(
            encoding="utf-8"
        )
    )


    # --------------------------------------------------------
    # Hard protocol guards
    # --------------------------------------------------------

    assert (
        cfg["train"]["batch_size"]
        == 4
    ), (
        f"Final72 A0 requires batch_size=4, "
        f"got {cfg['train']['batch_size']}"
    )

    # PORTABILITY PATCH 1/3: CSV identity is enforced by the SHA-256
    # split lock in train.py, independent of its absolute path.
    actual_train_csv = Path(cfg["paths"]["train_csv"]).expanduser().resolve()
    actual_val_csv = Path(cfg["paths"]["val_csv"]).expanduser().resolve()
    if not actual_train_csv.is_file() or not actual_val_csv.is_file():
        raise FileNotFoundError("Portable A3 train/val CSV not found")


    set_seed(
        SEED
    )


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    # --------------------------------------------------------
    # Same data / preprocessing as V1a.
    #
    # batch 里虽然还可能包含 raw/response/u_map，
    # 本 baseline 完全不使用。
    # --------------------------------------------------------

    # --------------------------------------------------------
    # A3 formal RGB + U-map dataset.
    #
    # IMPORTANT:
    # - same Train/Val CSV
    # - same RGB preprocessing
    # - same synchronized train hflip
    # - GT remains original 0~5
    # - U = uint16 / 65535
    # - no Response
    # --------------------------------------------------------

    # PORTABILITY PATCH 2/3: required globally encoded uint16 U-maps.
    umap_dir = Path(cfg["paths"]["umap_dir"]).expanduser().resolve()

    raw_dir = Path(
        cfg["paths"]["raw_dir"]
    ).resolve()

    train_dataset = (
        Final72RGBUMapDataset(
            actual_train_csv,
            raw_dir,
            umap_dir,
            training=True,
        )
    )

    val_dataset = (
        Final72RGBUMapDataset(
            actual_val_csv,
            raw_dir,
            umap_dir,
            training=False,
        )
    )

    workers = int(
        cfg["train"]["num_workers"]
    )

    common_loader = dict(
        num_workers=workers,
        pin_memory=bool(
            cfg["train"].get(
                "pin_memory",
                True,
            )
        ),
        persistent_workers=(
            bool(
                cfg["train"].get(
                    "persistent_workers",
                    False,
                )
            )
            and workers > 0
        ),
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=
            cfg["train"]["batch_size"],
        sampler=EpochShuffleSampler(
            train_dataset,
            SEED,
        ),
        shuffle=False,
        drop_last=bool(
            cfg["train"].get(
                "drop_last",
                True,
            )
        ),
        generator=
            torch.Generator()
            .manual_seed(SEED),
        **common_loader,
    )

    # Existing build_loaders() used val_batch_size=1.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        generator=
            torch.Generator()
            .manual_seed(SEED),
        **common_loader,
    )

    n_train = len(
        train_loader.dataset
    )

    n_val = len(
        val_loader.dataset
    )

    assert n_train == 4414, n_train
    assert n_val == 493, n_val


    # --------------------------------------------------------
    # Pure B0 model
    # --------------------------------------------------------

    ret_ckpt = resolve_ret_ckpt(
        cfg
    )

    model = RETFoundB15TaskQueryUMap4Head(
        checkpoint_path=ret_ckpt,
        image_size=384,
    ).to(
        device
    )


    backbone_trainable = sum(
        p.numel()
        for p in model.backbone.parameters()
        if p.requires_grad
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


    if backbone_trainable != 0:

        raise RuntimeError(
            "RETFound backbone is not frozen"
        )

    if trainable_params != 1185253:

        raise RuntimeError(
            "Expected exactly 1185253 trainable "
            f"parameters, got {trainable_params}"
        )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=BASE_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )


    ema = SimpleEMA(
        model,
        decay=EMA_DECAY,
    )


    # --------------------------------------------------------
    # Preflight print
    # --------------------------------------------------------

    print("=" * 100)
    print("A3 B15 TASKQUERY + U-MAP FINAL72 CONTROLLED")
    print("=" * 100)

    print("train             =", n_train)
    print("val               =", n_val)
    print("test accessed     = FALSE")

    print()
    print("architecture      = Frozen RETFound B15 + TaskQuery + U Spatial Interaction")
    print("trainable params  =", trainable_params)
    print("backbone trainable=", backbone_trainable)

    print()
    print("seed              =", SEED)
    print("selected block    = B15")
    print("batch             =", cfg["train"]["batch_size"])
    print("total_iters       =", TOTAL_ITERS)
    print("val_freq          =", VAL_FREQ)
    print("early_stop        = FALSE")

    print()
    print("optimizer         = Adam")
    print("base_lr           =", BASE_LR)
    print("milestones        =", MILESTONES)
    print("gamma             =", GAMMA)

    print()
    print("SmoothL1 beta     =", SMOOTH_L1_BETA)
    print("GT scale          = original 0~5 -> /5 in trainer")
    print("EMA decay         =", EMA_DECAY)
    print("AMP               = FALSE")

    print("=" * 100)


    metrics_file = (
        output_dir
        / "metrics.jsonl"
    )

    train_log = (
        output_dir
        / "train.log"
    )


    completed_steps = 0
    epoch = 0
    validation_count = 0

    best_mean_srcc = (
        -float("inf")
    )


    # ========================================================
    # TRAIN
    # ========================================================

    while (
        completed_steps
        <
        TOTAL_ITERS
    ):

        model.train()

        for batch in train_loader:

            if (
                completed_steps
                >=
                TOTAL_ITERS
            ):
                break


            lr = set_optimizer_lr(
                optimizer,
                completed_steps,
            )


            img = (
                batch["img"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            u_map = (
                batch["u_map"]
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            gt = (
                batch["gt"]
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )

            gt_mask = (
                batch["gt_mask"]
                .float()
                .to(
                    device,
                    non_blocking=True,
                )
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            pred = model(
                img,
                u_map,
            )


            loss, parts = (
                four_task_loss(
                    pred,
                    gt,
                    gt_mask,
                )
            )


            loss.backward()


            # Frozen backbone guard
            if any(
                p.grad is not None
                for p
                in model.backbone.parameters()
            ):
                raise RuntimeError(
                    "Frozen RETFound received gradients"
                )


            grad_sq = 0.0

            for p in model.parameters():

                if (
                    p.requires_grad
                    and
                    p.grad is not None
                ):

                    grad_sq += float(
                        p.grad.detach()
                        .float()
                        .square()
                        .sum()
                        .cpu()
                    )

            grad_norm = math.sqrt(
                grad_sq
            )


            optimizer.step()

            ema.update(
                model
            )

            completed_steps += 1


            # ------------------------------------------------
            # Train log
            # ------------------------------------------------

            if (
                completed_steps
                % PRINT_FREQ
                == 0
            ):

                log = {
                    "event":
                        "train",

                    "completed_steps":
                        completed_steps,

                    "epoch":
                        epoch,

                    "lr":
                        lr,

                    "loss_total":
                        float(
                            loss.detach().cpu()
                        ),

                    "loss_clarity":
                        float(
                            parts[
                                "loss_clarity"
                            ].detach().cpu()
                        ),

                    "loss_exposure":
                        float(
                            parts[
                                "loss_exposure"
                            ].detach().cpu()
                        ),

                    "loss_artifact":
                        float(
                            parts[
                                "loss_artifact"
                            ].detach().cpu()
                        ),

                    "loss_completeness":
                        float(
                            parts[
                                "loss_completeness"
                            ].detach().cpu()
                        ),

                    "grad_norm":
                        grad_norm,
                }


                line = json.dumps(
                    log,
                    ensure_ascii=False,
                )

                print(
                    "TRAIN " + line,
                    flush=True,
                )


                with train_log.open(
                    "a",
                    encoding="utf-8",
                ) as f:

                    f.write(
                        line + "\n"
                    )


            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            if (
                completed_steps
                % VAL_FREQ
                == 0
            ):

                metrics = validate(
                    model,
                    val_loader,
                    device,
                    ema,
                )

                metrics[
                    "event"
                ] = "validation"

                metrics[
                    "step"
                ] = completed_steps

                validation_count += 1


                print(
                    "VALIDATION "
                    +
                    json.dumps(
                        metrics,
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


                with metrics_file.open(
                    "a",
                    encoding="utf-8",
                ) as f:

                    f.write(
                        json.dumps(
                            metrics,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


                mean_srcc = float(
                    metrics[
                        "mean_SRCC"
                    ]
                )


                if (
                    mean_srcc
                    >
                    best_mean_srcc
                ):

                    best_mean_srcc = (
                        mean_srcc
                    )


                    best_metrics = dict(
                        metrics
                    )

                    best_metrics[
                        "best_step"
                    ] = completed_steps


                    (
                        output_dir
                        / "val_metrics_best_srcc.json"
                    ).write_text(
                        json.dumps(
                            best_metrics,
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )


                    save_checkpoint(
                        output_dir
                        / "best_srcc.pt",

                        model,
                        ema,
                        optimizer,

                        completed_steps,
                        epoch,

                        best_metrics,
                    )


                model.train()


        epoch += 1


    # ========================================================
    # FINAL
    # ========================================================

    if (
        completed_steps
        != TOTAL_ITERS
    ):
        raise RuntimeError(
            f"Expected {TOTAL_ITERS}, "
            f"got {completed_steps}"
        )


    expected_val_count = (
        TOTAL_ITERS
        // VAL_FREQ
    )

    if (
        validation_count
        != expected_val_count
    ):
        raise RuntimeError(
            f"Expected {expected_val_count} validations, "
            f"got {validation_count}"
        )


    save_checkpoint(
        output_dir
        / "latest.pt",

        model,
        ema,
        optimizer,

        completed_steps,
        epoch,
        None,
    )


    manifest = {
        "status":
            "complete_total_iters",

        "completed_steps":
            completed_steps,

        "total_iters":
            TOTAL_ITERS,

        "validation_count":
            validation_count,

        "early_stopping":
            False,

        "best_mean_SRCC":
            best_mean_srcc,

        "train_samples":
            n_train,

        "val_samples":
            n_val,

        "test_access":
            False,

        "architecture":
            "Frozen RETFound B15 + Four Task Queries + U-map Cross-Attention + Gated Residual Fusion",
    }


    (
        output_dir
        / "run_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    print()
    print("=" * 100)
    print("B0 FIXED-LR TRAINING FINISHED")
    print("=" * 100)
    print(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

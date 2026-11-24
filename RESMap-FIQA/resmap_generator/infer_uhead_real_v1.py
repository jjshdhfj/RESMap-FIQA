#!/usr/bin/env python3
"""Infer V1 restoration uncertainty on real fundus images without HQ targets.

For each matched image/mask pair, this script:
  1. resizes image and mask to --image_size,
  2. runs the frozen public GFENet once,
  3. constructs the 8-channel V1 input,
  4. predicts log-scale and uncertainty scale,
  5. saves enhanced image, response map and raw uncertainty products.

No high-quality reference image is required. ``uncertainty_raw/*.npy`` is set
exactly to zero outside the original (non-eroded) fundus mask. Because log(0)
is undefined, ``log_scale_raw/*.npy`` is set to log_scale_min outside the mask.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from uhead_v1 import (
    UncertaintyHeadV1,
    build_uhead_input_v1,
    erode_binary_mask,
    rgb_to_luminance,
)


DEFAULT_PROJECT_ROOT = Path(
    "/home/ly/jbk/增强模型/Annotation-free-Fundus-Image-Enhancement-main"
)
DEFAULT_GFENET_CHECKPOINT = (
    DEFAULT_PROJECT_ROOT / "checkpoints/gfenet_public/latest_net_G.pth"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer frozen-GFENet V1 uncertainty on real images without HQ."
    )
    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--mask_dir", type=Path, required=True)
    parser.add_argument(
        "--gfenet_checkpoint", type=Path, default=DEFAULT_GFENET_CHECKPOINT
    )
    parser.add_argument("--uhead_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0, help="Use -1 for CPU")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--mask_erode_kernel", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="0 means all matched pairs")
    parser.add_argument(
        "--strict_pairs",
        action="store_true",
        help="Fail if any image lacks a same-stem mask or any extra mask exists.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save_response_u16",
        action="store_true",
        help="Save fixed-scale uint16 luminance response PNGs.",
    )
    parser.add_argument(
        "--save_uncertainty_u16",
        action="store_true",
        help="Also save globally encoded uncertainty scale as uint16 PNG.",
    )
    return parser.parse_args()


def collect_by_stem(folder: Path) -> Dict[str, Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    result: Dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in result:
            raise RuntimeError(
                "Duplicate stem with different files:\n"
                f"{result[path.stem]}\n{path}"
            )
        result[path.stem] = path.resolve()
    if not result:
        raise RuntimeError(f"No supported images found in: {folder}")
    return result


def pair_images_and_masks(
    image_dir: Path,
    mask_dir: Path,
    strict: bool,
    limit: int,
) -> Tuple[List[Tuple[str, Path, Path]], List[str], List[str]]:
    images = collect_by_stem(image_dir)
    masks = collect_by_stem(mask_dir)
    missing_masks = sorted(set(images) - set(masks))
    extra_masks = sorted(set(masks) - set(images))
    if strict and (missing_masks or extra_masks):
        raise RuntimeError(
            f"Pair check failed: missing_masks={len(missing_masks)}, "
            f"extra_masks={len(extra_masks)}"
        )
    stems = sorted(set(images) & set(masks))
    if limit > 0:
        stems = stems[:limit]
    pairs = [(stem, images[stem], masks[stem]) for stem in stems]
    if not pairs:
        raise RuntimeError("No same-stem image/mask pairs were found")
    return pairs, missing_masks, extra_masks


class RealFundusDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, Path, Path]], image_size: int) -> None:
        self.pairs = list(pairs)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, object]:
        stem, image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            original_size = image.size
            if image.size != (self.image_size, self.image_size):
                image = image.resize(
                    (self.image_size, self.image_size), Image.BICUBIC
                )
            image_array = np.asarray(image, dtype=np.float32) / 255.0

        with Image.open(mask_path) as mask_file:
            mask = mask_file.convert("L")
            if mask.size != (self.image_size, self.image_size):
                mask = mask.resize(
                    (self.image_size, self.image_size), Image.NEAREST
                )
            mask_array = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)

        if float(mask_array.mean()) <= 0.01:
            raise RuntimeError(f"Mask is nearly empty: {mask_path}")

        image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask_array[None, :, :]).float()
        image_tensor = image_tensor * mask_tensor

        return {
            "stem": stem,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_width": int(original_size[0]),
            "original_height": int(original_size[1]),
            "image_01": image_tensor,
            "image_n11": image_tensor * 2.0 - 1.0,
            "mask": mask_tensor,
        }


def prepare_output(output_dir: Path, overwrite: bool, resume: bool) -> Dict[str, Path]:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        elif not resume:
            raise RuntimeError(
                f"Output directory is not empty: {output_dir}\n"
                "Use --resume or --overwrite."
            )
    directories = {
        "enhanced": output_dir / "enhanced",
        "uncertainty_raw": output_dir / "uncertainty_raw",
        "log_scale_raw": output_dir / "log_scale_raw",
        "uncertainty_vis": output_dir / "uncertainty_vis",
        "response_u16": output_dir / "response_luma_u16",
        "uncertainty_u16": output_dir / "uncertainty_scale_u16",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def completed_stems(output_dirs: Dict[str, Path]) -> set:
    completed = set()
    for path in output_dirs["uncertainty_raw"].glob("*.npy"):
        stem = path.stem
        required = [
            output_dirs["enhanced"] / f"{stem}.png",
            output_dirs["log_scale_raw"] / f"{stem}.npy",
        ]
        if all(item.is_file() for item in required):
            completed.add(stem)
    return completed


def build_frozen_gfenet(args: argparse.Namespace):
    project_root = args.project_root.resolve()
    checkpoint = args.gfenet_checkpoint.resolve()
    if not (project_root / "models/gfenet_model.py").is_file():
        raise FileNotFoundError(f"Not a GFENet project root: {project_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GFENet checkpoint does not exist: {checkpoint}")
    if not checkpoint.name.endswith("_net_G.pth"):
        raise RuntimeError(
            "GFENet checkpoint must follow <epoch>_net_G.pth, e.g. latest_net_G.pth"
        )

    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --gpu -1 for CPU")
        torch.cuda.set_device(args.gpu)
        gpu_ids = [args.gpu]
    else:
        gpu_ids = []

    os.chdir(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from models.gfenet_model import GFENetModel  # type: ignore

    epoch = checkpoint.name[: -len("_net_G.pth")]
    options = SimpleNamespace(
        gpu_ids=gpu_ids,
        isTrain=False,
        checkpoints_dir=str(checkpoint.parent.parent),
        name=checkpoint.parent.name,
        preprocess="none",
        direction="AtoB",
        input_nc=3,
        output_nc=3,
        ngf=64,
        netG="unet_gfe_net",
        norm="instance",
        no_dropout=True,
        init_type="normal",
        init_gain=0.02,
        num_of_filter=3,
        filters_width_list=[9, 19, 29],
        nsig_list=[5.0, 9.0, 13.0],
        sub_low_ratio=1.0,
        epoch=epoch,
        load_iter=0,
        verbose=False,
    )
    model = GFENetModel(options)
    model.setup(options)
    model.eval()
    for parameter in model.netG.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.netG.parameters()):
        raise RuntimeError("GFENet was not fully frozen")
    return model


def tensor_rgb_to_u8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def save_map_u16(path: Path, value_01: np.ndarray) -> None:
    array = np.clip(np.rint(value_01 * 65535.0), 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), array, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"Failed to save uint16 PNG: {path}")


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 256 != 0:
        raise ValueError("--image_size must be a positive multiple of 256")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    args.project_root = args.project_root.resolve()
    args.image_dir = args.image_dir.resolve()
    args.mask_dir = args.mask_dir.resolve()
    args.gfenet_checkpoint = args.gfenet_checkpoint.resolve()
    args.uhead_checkpoint = args.uhead_checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()

    pairs, missing_masks, extra_masks = pair_images_and_masks(
        args.image_dir,
        args.mask_dir,
        args.strict_pairs,
        args.limit,
    )
    output_dirs = prepare_output(
        args.output_dir, overwrite=args.overwrite, resume=args.resume
    )
    done = completed_stems(output_dirs) if args.resume else set()
    pending_pairs = [pair for pair in pairs if pair[0] not in done]

    print(f"images with matched masks: {len(pairs)}")
    print(f"images missing masks: {len(missing_masks)}")
    print(f"extra masks: {len(extra_masks)}")
    print(f"already completed: {len(pairs) - len(pending_pairs)}")
    print(f"pending: {len(pending_pairs)}")
    if missing_masks:
        print("missing mask examples:", missing_masks[:20])
    if extra_masks:
        print("extra mask examples:", extra_masks[:20])
    if not pending_pairs:
        print("Nothing to process.")
        return

    model = build_frozen_gfenet(args)
    device = model.device

    uhead_payload = torch.load(args.uhead_checkpoint, map_location=device)
    config = uhead_payload.get("config", {})
    log_scale_min = float(config.get("log_scale_min", -7.0))
    log_scale_max = float(config.get("log_scale_max", 0.0))
    net_u = UncertaintyHeadV1(
        in_channels=8,
        base_channels=int(config.get("base_channels", 32)),
        log_scale_min=log_scale_min,
        log_scale_max=log_scale_max,
    ).to(device)
    net_u.load_state_dict(uhead_payload["uhead"], strict=True)
    net_u.eval()

    dataset = RealFundusDataset(pending_pairs, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    summary_rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            image_01 = batch["image_01"].to(device, non_blocking=True)
            image_n11 = batch["image_n11"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            image_paths = list(batch["image_path"])

            model.set_input(
                {
                    "TA": image_n11,
                    "T_mask": mask,
                    "TA_path": image_paths,
                }
            )
            model.test()
            enhanced = ((model.fake_TB + 1.0) * 0.5).clamp(0.0, 1.0) * mask
            image_01 = image_01 * mask
            response = torch.abs(
                rgb_to_luminance(enhanced) - rgb_to_luminance(image_01)
            ) * mask
            log_scale = net_u(
                build_uhead_input_v1(
                    image_01, enhanced, mask, response_luma=response
                )
            )
            uncertainty = torch.exp(log_scale)
            valid = erode_binary_mask(mask, args.mask_erode_kernel)

            for index in range(image_01.shape[0]):
                stem = str(batch["stem"][index])
                fov = mask[index, 0].cpu().numpy() > 0.5
                valid_np = valid[index, 0].cpu().numpy() > 0.5
                u_np = uncertainty[index, 0].cpu().numpy().astype(np.float32)
                s_np = log_scale[index, 0].cpu().numpy().astype(np.float32)
                r_np = response[index, 0].cpu().numpy().astype(np.float32)

                # Required behavior: raw uncertainty is exactly zero outside FOV.
                u_np[~fov] = 0.0
                r_np[~fov] = 0.0
                # log(0) is undefined; retain the model lower bound outside FOV.
                s_np[~fov] = log_scale_min

                Image.fromarray(tensor_rgb_to_u8(enhanced[index]), mode="RGB").save(
                    output_dirs["enhanced"] / f"{stem}.png"
                )
                np.save(output_dirs["uncertainty_raw"] / f"{stem}.npy", u_np)
                np.save(output_dirs["log_scale_raw"] / f"{stem}.npy", s_np)

                if args.save_response_u16:
                    save_map_u16(output_dirs["response_u16"] / f"{stem}.png", r_np)
                if args.save_uncertainty_u16:
                    save_map_u16(output_dirs["uncertainty_u16"] / f"{stem}.png", u_np)

                valid_values = u_np[valid_np]
                high = float(np.percentile(valid_values, 99)) if valid_values.size else 1.0
                vis = np.zeros_like(u_np, dtype=np.uint8)
                if high > 0:
                    vis[valid_np] = np.clip(
                        u_np[valid_np] / high * 255.0, 0, 255
                    ).astype(np.uint8)
                Image.fromarray(vis, mode="L").save(
                    output_dirs["uncertainty_vis"] / f"{stem}.png"
                )

                denominator = max(float(valid_np.sum()), 1.0)
                summary_rows.append(
                    {
                        "stem": stem,
                        "image_path": str(batch["image_path"][index]),
                        "mask_path": str(batch["mask_path"][index]),
                        "original_width": int(batch["original_width"][index]),
                        "original_height": int(batch["original_height"][index]),
                        "model_size": args.image_size,
                        "mean_uncertainty": float((u_np * valid_np).sum() / denominator),
                        "p95_uncertainty": float(
                            np.percentile(u_np[valid_np], 95)
                            if valid_np.any()
                            else 0.0
                        ),
                        "mean_response_luma": float((r_np * valid_np).sum() / denominator),
                    }
                )
                print(stem)

    summary_path = args.output_dir / "summary.csv"
    fieldnames = [
        "stem",
        "image_path",
        "mask_path",
        "original_width",
        "original_height",
        "model_size",
        "mean_uncertainty",
        "p95_uncertainty",
        "mean_response_luma",
    ]
    mode = "a" if args.resume and summary_path.is_file() else "w"
    with summary_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w" or summary_path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(summary_rows)

    run_summary = {
        "project_root": str(args.project_root),
        "image_dir": str(args.image_dir),
        "mask_dir": str(args.mask_dir),
        "gfenet_checkpoint": str(args.gfenet_checkpoint),
        "uhead_checkpoint": str(args.uhead_checkpoint),
        "matched_pairs_selected": len(pairs),
        "processed_now": len(summary_rows),
        "missing_mask_count": len(missing_masks),
        "extra_mask_count": len(extra_masks),
        "image_size": args.image_size,
        "raw_uncertainty_outside_mask": 0.0,
        "raw_log_scale_outside_mask": log_scale_min,
        "requires_hq": False,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

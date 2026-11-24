#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RESMap-FIQA A3 inference on RGB fundus images with paired FOV masks.

Two modes:
  (1) Generate uint16 RESMaps using the copied GFENet + U-head V1 sources,
      then predict four A3 image-quality scores.
  (2) Reuse already generated uint16 RESMaps and only run A3 inference.

No ground-truth labels, study CSVs, original research server paths, or training
code are needed. The RGB preprocessing and EMA loading are matched to the
verified A3 test and prediction-comparison scripts.

Examples:
  python inference_resmap_fiqa.py --release-dir ./RESMap-FIQA-Code-with-RESMap-CANDIDATE \
    --image-dir /path/images --mask-dir /path/masks \
    --gfenet-checkpoint /path/latest_net_G.pth \
    --uhead-checkpoint /path/best_uhead_v1_snapshot_512.pth \
    --retfound-checkpoint /path/RETFound_mae_natureCFP.pth \
    --a3-checkpoint /path/best_srcc.pt \
    --output-dir /path/inference_output

  python inference_resmap_fiqa.py --release-dir ./RESMap-FIQA-Code-with-RESMap-CANDIDATE \
    --image-dir /path/images --resmap-dir /path/uncertainty_scale_u16 \
    --retfound-checkpoint /path/RETFound_mae_natureCFP.pth \
    --a3-checkpoint /path/best_srcc.pt \
    --output-dir /path/inference_output
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

EXTENSIONS = ('.jpeg', '.jpg', '.png', '.bmp', '.tif', '.tiff')
TASKS = ('clarity', 'exposure', 'artifact', 'completeness')


def index_images(folder: Path, extensions, label: str):
    if not folder.is_dir():
        raise NotADirectoryError('%s directory missing: %s' % (label, folder))
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in result:
                raise ValueError('%s has duplicate stem %r: %s / %s' %
                                 (label, path.stem, result[path.stem], path))
            result[path.stem] = path.resolve()
    if not result:
        raise ValueError('No %s files found: %s' % (label, folder))
    return result


def validate_paired(left, right, left_name, right_name):
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    if missing or extra:
        raise ValueError('%s and %s have unmatched file stems. Missing %s: %s; '
                         'extra %s: %s' % (left_name, right_name, right_name,
                                           missing[:12], right_name, extra[:12]))
    return sorted(left)


def restore_ema(model, path: Path):
    # Precisely the trainable-shadow semantics used in the verified A3-only test.
    checkpoint = torch.load(str(path), map_location='cpu')
    shadow = checkpoint.get('ema', {}).get('shadow')
    if not isinstance(shadow, dict) or not shadow:
        raise RuntimeError('A3 checkpoint missing ema.shadow: ' + str(path))
    named = dict(model.named_parameters())
    expected = {name for name, p in named.items() if p.requires_grad}
    missing = expected - set(shadow)
    unknown = set(shadow) - set(named)
    if missing or unknown:
        raise RuntimeError('A3 EMA mismatch: missing=%s unknown=%s' %
                           (sorted(missing)[:10], sorted(unknown)[:10]))
    with torch.no_grad():
        for name, weight in shadow.items():
            named[name].copy_(weight.to(device=named[name].device,
                                        dtype=named[name].dtype))
    return checkpoint.get('completed_steps')


def preprocess_pair(rgb_path, resmap_path, arq_data, read_umap_u16, mean, std):
    rgb = arq_data._resize(
        arq_data._read_rgb(rgb_path), arq_data.RISK_SIZE, 'bilinear')
    rgb = arq_data._resize(rgb, arq_data.RETFOUND_SIZE, 'bilinear')
    rgb = ((rgb - mean) / std).unsqueeze(0).contiguous()
    u = read_umap_u16(resmap_path, rgb_path.stem)
    if tuple(u.shape[-2:]) != tuple(arq_data.RISK_SIZE):
        u = arq_data._resize(u, arq_data.RISK_SIZE, 'bilinear')
    u = arq_data._resize(u, arq_data.RETFOUND_SIZE, 'bilinear')
    u = u.unsqueeze(0).contiguous()
    if tuple(rgb.shape) != (1, 3, 384, 384) or tuple(u.shape) != (1, 1, 384, 384):
        raise RuntimeError('Unexpected A3 image/map shapes: %s / %s' %
                           (tuple(rgb.shape), tuple(u.shape)))
    return rgb, u


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--release-dir', type=Path, default=Path(__file__).resolve().parent,
                    help='Root with arq/, models_vit.py and resmap_generator/')
    ap.add_argument('--image-dir', type=Path, required=True)
    ap.add_argument('--mask-dir', type=Path, help='Matched original FOV masks. Required unless --resmap-dir is used.')
    ap.add_argument('--resmap-dir', type=Path,
                    help='Use precomputed single-channel uint16 RESMap PNGs instead of GFENet generation.')
    ap.add_argument('--gfenet-checkpoint', type=Path)
    ap.add_argument('--uhead-checkpoint', type=Path)
    ap.add_argument('--retfound-checkpoint', type=Path, required=True)
    ap.add_argument('--a3-checkpoint', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--generator-gpu', type=int, default=-1,
                    help='-1 = CPU (reproduced formal uint16 maps exactly in validation); 0 = CUDA:0')
    ap.add_argument('--a3-device', choices=('auto','cpu','cuda'), default='auto')
    ap.add_argument('--overwrite-results', action='store_true',
                    help='Permit replacing an existing predictions.csv, but never overwrite external originals.')
    args = ap.parse_args()
    for key in ('release_dir', 'image_dir', 'mask_dir', 'resmap_dir',
                'gfenet_checkpoint', 'uhead_checkpoint', 'retfound_checkpoint',
                'a3_checkpoint', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    release = args.release_dir
    if not (release / 'arq/data.py').is_file():
        ap.error('--release-dir does not contain arq/data.py: %s' % release)
    if not (release / 'resmap_generator/infer_uhead_real_v1.py').is_file() and args.resmap_dir is None:
        ap.error('Missing copied RESMap generator under --release-dir')
    for label, path in (('RETFound checkpoint',args.retfound_checkpoint),
                        ('A3 checkpoint',args.a3_checkpoint)):
        if not path.is_file():
            ap.error('%s missing: %s' % (label, path))
    images = index_images(args.image_dir, EXTENSIONS, 'RGB')
    if args.resmap_dir is None:
        for label, path in (('FOV masks',args.mask_dir),
                            ('GFENet checkpoint',args.gfenet_checkpoint),
                            ('U-head checkpoint',args.uhead_checkpoint)):
            if path is None or not path.is_file() and label != 'FOV masks':
                ap.error('%s required and must exist when generating RESMaps' % label)
        masks = index_images(args.mask_dir, EXTENSIONS, 'FOV mask')
        validate_paired(images,masks,'RGB','FOV mask')
    else:
        if not args.resmap_dir.is_dir():
            ap.error('--resmap-dir does not exist: %s' % args.resmap_dir)

    results_csv = args.output_dir / 'predictions.csv'
    if results_csv.exists() and not args.overwrite_results:
        ap.error('Refusing to overwrite existing predictions.csv: %s (use --overwrite-results)' % results_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resmap_dir is None:
        gen_output = args.output_dir / 'generated'
        if gen_output.exists():
            ap.error('Refusing to reuse/overwrite generated RESMap output: %s. Choose a new --output-dir.' % gen_output)
        gen_script = release / 'resmap_generator/infer_uhead_real_v1.py'
        cmd = [sys.executable, '-u', str(gen_script),
               '--project_root', str(release / 'resmap_generator'),
               '--image_dir', str(args.image_dir), '--mask_dir', str(args.mask_dir),
               '--gfenet_checkpoint', str(args.gfenet_checkpoint),
               '--uhead_checkpoint', str(args.uhead_checkpoint),
               '--output_dir', str(gen_output), '--image_size', '512',
               '--gpu', str(args.generator_gpu), '--batch_size', '1', '--num_workers', '0',
               '--save_uncertainty_u16', '--strict_pairs']
        print('[GENERATE] Using copied GFENet + U-head V1; generator GPU:', args.generator_gpu, flush=True)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        subprocess.run(cmd, cwd=str(release / 'resmap_generator'), env=env, check=True)
        maps = gen_output / 'uncertainty_scale_u16'
    else:
        maps = args.resmap_dir
        print('[RESMAP] Reading precomputed uint16 maps:',maps, flush=True)
    u_images = index_images(maps, ('.png',), 'RESMap')
    stems = validate_paired(images, u_images, 'RGB', 'RESMap')

    if args.a3_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        if args.a3_device == 'cuda' and not torch.cuda.is_available():
            ap.error('CUDA unavailable; set --a3-device cpu')
        device = torch.device(args.a3_device)
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(release))
    from arq import data as arq_data
    from arq.final72_rgb_umap_dataset import read_umap_u16
    from arq.retfound_b15_taskquery_umap_4head import RETFoundB15TaskQueryUMap4Head
    source = Path(arq_data.__file__).resolve()
    try:
        source.relative_to(release)
    except ValueError:
        raise RuntimeError('Imported original project arq instead of release: %s' % source)
    model = RETFoundB15TaskQueryUMap4Head(
        checkpoint_path=str(args.retfound_checkpoint), image_size=384).to(device)
    step = restore_ema(model, args.a3_checkpoint)
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 1185253:
        raise RuntimeError('A3 trainable param mismatch: %s (expected 1185253)' % trainable)
    print('[A3] Source:', source, '| trainable:', trainable, '| EMA step:', step, '| device:', device, flush=True)
    mean = torch.tensor(arq_data.IMAGENET_MEAN, dtype=torch.float32).view(3,1,1)
    std = torch.tensor(arq_data.IMAGENET_STD, dtype=torch.float32).view(3,1,1)
    records=[]
    for stem in stems:
        rgb,u = preprocess_pair(images[stem],u_images[stem],arq_data,read_umap_u16,mean,std)
        with torch.no_grad():
            scores = (model(rgb.to(device),u.to(device)).detach().float().cpu().numpy().reshape(-1)*5.0)
        if scores.shape != (4,) or not np.isfinite(scores).all():
            raise RuntimeError('A3 returned invalid 4 scores for: '+stem)
        row={'image':images[stem].name, 'resmap':str(u_images[stem]),
             **{task:float(score) for task, score in zip(TASKS,scores)}}
        records.append(row)
        print('[PREDICT] %-25s %s' % (stem, '  '.join('%s=%.6f' % (k,row[k]) for k in TASKS)), flush=True)
    with results_csv.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=('image','resmap')+TASKS)
        writer.writeheader()
        for record in records: writer.writerow(record)
    report={
        'source_release':str(release), 'image_dir':str(args.image_dir),
        'resmap_dir':str(maps),'a3_checkpoint':str(args.a3_checkpoint),
        'retfound_checkpoint':str(args.retfound_checkpoint),'a3_device':str(device),
        'generator_gpu':None if args.resmap_dir is not None else args.generator_gpu,
        'a3_ema_completed_steps':step,'trainable_a3_parameters':trainable,
        'image_count':len(records),'predictions_csv':str(results_csv),
    }
    (args.output_dir/'inference_run.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('[PASS] %d images inferred. Results: %s' % (len(records),results_csv),flush=True)

if __name__=='__main__':
    main()

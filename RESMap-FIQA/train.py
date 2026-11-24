#!/usr/bin/env python3
"""Portable, protocol-guarded launcher for the unchanged A3 training algorithm.

This launcher resolves paths, verifies the locked CSV fingerprints, performs a
non-training data preflight, and invokes tools/train_a3_portable.py. It does
not modify the validated original training script.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REQUIRED_PATHS = ('train_csv', 'val_csv', 'raw_dir', 'umap_dir', 'ret_ckpt')
EXPECTED_COUNTS = {'train_csv': 4414, 'val_csv': 493}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def abs_path(value, base):
    p = Path(os.path.expandvars(str(value))).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def load_csv(path):
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            with Path(path).open('r', encoding=encoding, newline='') as f:
                rows = list(csv.DictReader(f))
            return rows
        except UnicodeError:
            continue
    raise RuntimeError('Cannot decode CSV as UTF-8 or GB18030: ' + str(path))


def prepare_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict) or not isinstance(cfg.get('paths'), dict) or not isinstance(cfg.get('train'), dict):
        raise RuntimeError('YAML needs paths and train mappings.')
    for key in REQUIRED_PATHS:
        value = cfg['paths'].get(key)
        if not value or str(value).startswith('/path/to/'):
            raise RuntimeError('Missing or placeholder config paths.%s' % key)
        p = abs_path(value, config_path.parent)
        if not (p.is_file() if key.endswith('_csv') or key == 'ret_ckpt' else p.is_dir()):
            raise FileNotFoundError('%s: %s' % (key, p))
        cfg['paths'][key] = str(p)
    # The historical trainer uses exactly this protocol.
    if cfg['train'].get('batch_size') != 4:
        raise RuntimeError('A3 historical protocol requires train.batch_size=4')
    if 'num_workers' not in cfg['train']:
        raise RuntimeError('train.num_workers is required')
    return cfg


def preflight(cfg, lock_path, allow_unlocked=False, check_samples=True):
    lock = None
    if lock_path is not None:
        path = Path(lock_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        lock = json.loads(path.read_text(encoding='utf-8'))
        if lock.get('protocol') != 'A3-Final72-seed3407-file-sha256':
            raise RuntimeError('Unrecognized split lock format')
    elif not allow_unlocked:
        raise RuntimeError('Official reproduction requires --split-lock. For non-locked experiments use --allow-unlocked explicitly.')
    info = {}
    sets = {}
    for key in ('train_csv', 'val_csv'):
        path = Path(cfg['paths'][key])
        rows = load_csv(path)
        if not rows or 'file_name' not in rows[0]:
            raise RuntimeError('%s must have file_name column and nonempty rows' % key)
        if len(rows) != EXPECTED_COUNTS[key]:
            raise RuntimeError('%s: expected %s rows, got %s' % (key, EXPECTED_COUNTS[key], len(rows)))
        names = [Path(str(r['file_name'])).stem for r in rows]
        if len(set(names)) != len(names):
            raise RuntimeError('Duplicate image filename stems in %s' % key)
        sets[key] = set(names)
        digest = sha256(path)
        if lock:
            expected = lock.get('files', {}).get(key, {})
            if digest != expected.get('sha256') or len(rows) != expected.get('rows'):
                raise RuntimeError('%s: split lock fingerprint/count mismatch (DO NOT train as locked protocol)' % key)
        info[key] = {'sha256': digest, 'rows': len(rows), 'stem_count': len(names)}
    if sets['train_csv'] & sets['val_csv']:
        raise RuntimeError('Train/val image filename stems overlap')
    # Do not train on test files. Test lock is optional and only used to identify the held-out split.
    if lock and lock.get('files', {}).get('test_csv'):
        info['test_lock_sha256'] = lock['files']['test_csv']['sha256']
    if lock and int(os.getenv('RUN_SEED','3407')) != 3407:
        raise RuntimeError('Original locked split protocol requires RUN_SEED=3407')
    info['locked_split_checked'] = bool(lock)
    info['train_batch_size'] = cfg['train']['batch_size']
    info['seed'] = int(os.getenv('RUN_SEED', '3407'))
    if check_samples:
        from arq.final72_rgb_umap_dataset import Final72RGBUMapDataset
        tr = Final72RGBUMapDataset(cfg['paths']['train_csv'], cfg['paths']['raw_dir'], cfg['paths']['umap_dir'], training=True)
        va = Final72RGBUMapDataset(cfg['paths']['val_csv'], cfg['paths']['raw_dir'], cfg['paths']['umap_dir'], training=False)
        if len(tr) != 4414 or len(va) != 493:
            raise RuntimeError('Dataset size mismatch: %s/%s' % (len(tr), len(va)))
        a, b = tr[0], va[0]
        for s in (a, b):
            if tuple(s['img'].shape) != (3, 384, 384) or tuple(s['u_map'].shape) != (1, 384, 384):
                raise RuntimeError('RGB/RESMap preprocess shape mismatch')
        info['first_sample_loaded'] = True
        print('[PASS] Dataset first sample RGB (3,384,384) / uint16 RESMap (1,384,384)', flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--split-lock', type=Path)
    ap.add_argument('--allow-unlocked', action='store_true')
    ap.add_argument('--preflight', action='store_true', help='Check inputs and exit without training')
    ap.add_argument('--smoke-step', action='store_true', help='One real optimizer+EMA step, no official checkpoints')
    ap.add_argument('--skip-first-sample', action='store_true')
    args = ap.parse_args()
    if args.split_lock and args.allow_unlocked:
        ap.error('Do not combine --split-lock and --allow-unlocked')
    root = Path(__file__).resolve().parent
    output = args.output.expanduser().resolve()
    if (output / 'best_srcc.pt').exists() or (output / 'latest.pt').exists():
        raise RuntimeError('Refusing to reuse an output directory with A3 checkpoints: ' + str(output))
    cfg = prepare_config(args.config)
    info = preflight(cfg, args.split_lock, args.allow_unlocked, check_samples=not args.skip_first_sample)
    print('[PASS] Portable preflight:', json.dumps(info, ensure_ascii=False), flush=True)
    if args.preflight:
        print('[PREFLIGHT ONLY] No training update performed.')
        return
    # Smoke is performed in-memory; no optimizer/EMA checkpoint is written.
    if args.smoke_step:
        import tempfile
        with tempfile.TemporaryDirectory(prefix='a3_portable_smoke_') as temp:
            path = Path(temp) / 'resolved.yaml'
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
            cmd = [sys.executable, '-u', str(root / 'tools/smoke_train_portable.py'), '--config', str(path)]
            subprocess.run(cmd, cwd=str(root), check=True)
        return
    if not (root / 'tools/train_a3_portable.py').is_file():
        raise FileNotFoundError(root / 'tools/train_a3_portable.py')
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / 'portable_resolved_config.yaml'
    if resolved.exists():
        raise FileExistsError('Refusing to overwrite pre-existing resolved config: ' + str(resolved))
    resolved.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding='utf-8')
    (output / 'portable_preflight.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[TRAIN] Using paths from resolved portable configuration. Original A3 code remains intact.', flush=True)
    cmd = [sys.executable, '-u', str(root / 'tools/train_a3_portable.py'), '--config', str(resolved), '--output', str(output)]
    subprocess.run(cmd, cwd=str(root), check=True)


if __name__ == '__main__':
    main()

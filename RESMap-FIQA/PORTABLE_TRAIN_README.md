# Portable A3 training and environment audit

The previously validated original trainer is unchanged:
`tools/train_a3_b15_taskquery_umap_final72.py`.
The new `tools/train_a3_portable.py` is a **separately derived copy** with only:
(i) path equality guards replaced with configuration-resolved CSV paths,
(ii) hardcoded RESMap directory replaced with `paths.umap_dir`, and
(iii) the RETFound fallback absolute path disabled. Its forward/backward/optimizer,
EMA, LR, seed, batch size, loss, validation and total-iteration code is unchanged.
It is **not** asserted numerically equivalent until the server training smoke
comparison has passed.

## Config

Edit `configs/portable_a3.yaml`. Relative file paths are relative to the YAML
file, not the shell's current directory. Mandatory keys:

```
paths: {train_csv: ..., val_csv: ..., raw_dir: ..., umap_dir: ..., ret_ckpt: ...}
train: {batch_size: 4, num_workers: 0, pin_memory: true, drop_last: true, persistent_workers: false}
```

The correct `umap_dir` contains **uint16 PNG** maps, encoded on the fixed
0..65535 scale; never use `uncertainty_vis` or per-image min-max maps.

## Locked split

A SHA-256 split manifest `configs/locked_split_sha256.json` is generated at
packaging time **only if** the builder sees the original server-side
train/val CSVs. This records their full file bytes and 4414/493 rows; it is
not a fresh clinical-label or patient-level leakage audit. If the paths are
relocated, the CSV bytes must remain identical for hash verification to pass.
For other research partitions, `--allow-unlocked` explicitly bypasses the
file SHA guard but the historical sample counts and batch size remain fixed.
Such runs should not be reported as the paper's locked experiment.

## Commands

From the extracted release root:

```bash
# First inspect requirements without changing your environment.
python environment_audit.py --report ./environment_audit.json

# Confirm paths, split fingerprints, 4414/493 samples and the first RGB/U-map
# sample WITHOUT training.
RUN_SEED=3407 python train.py \
  --config configs/portable_a3.yaml \
  --split-lock configs/locked_split_sha256.json \
  --output ./runs/a3_portable_trial \
  --preflight

# ONE real forward/backward/Adam/EMA update. Does not save an A3 checkpoint.
CUDA_VISIBLE_DEVICES=0 RUN_SEED=3407 python train.py \
  --config configs/portable_a3.yaml \
  --split-lock configs/locked_split_sha256.json \
  --output ./runs/a3_portable_trial \
  --smoke-step

# Only after confirming the above. This launches the full ORIGINAL protocol
# length (165450 updates); outputs in a NEW empty directory.
CUDA_VISIBLE_DEVICES=0 RUN_SEED=3407 python train.py \
  --config configs/portable_a3.yaml \
  --split-lock configs/locked_split_sha256.json \
  --output ./runs/a3_portable_full
```

If `locked_split_sha256.json` was not generated, use the original source CSVs
and regenerate a reference fingerprint in your private environment; do not
substitute arbitrary CSVs and declare them equivalent. The builder outputs
its own status explaining whether the lock is present.

## Environment

`environment_audit.py` checks imports of torch, torchvision, timm,
scikit-learn, scipy, opencv, Pillow, PyYAML and filelock; runs `pip check`;
records version, CUDA/GPU and package paths. **It does not modify packages or
prove an isolated clean install**. If `filelock` is missing, in the original
`basiqa` environment (after preserving package versions) use a reachable
mirror, excluding the known broken NVIDIA index:

```bash
PIP_CONFIG_FILE=/dev/null python -m pip install --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple filelock
```

Do not bulk-upgrade CUDA, PyTorch or timm just to solve a missing dependency.
Check and document actual installed versions first. A genuine cross-machine
clean install is a **separate verification**.

## Important limitations

- README documents the historical A3 trainer, but the original resolved YAML
  from the historical run was not supplied in the release. New template values
  are safe placeholders, **not evidence of the original YAML**.
- Only the original training/test code and inference/RESMap checks are previously
  validated; the portable derivative requires a real server smoke update.
- Redistribution/ownership review is documented in `THIRD_PARTY_RIGHTS.md`.
- Example data and third-party GFENet source are not cleared for public
  redistribution. The new archive's name must retain `INTERNAL`.

# RESMap-FIQA — code distribution candidate

**Publication status:** NOT YET CLEARED. Confirm code ownership / derivative
licensing and whether other third-party source files may be redistributed.

This package retains the previously tested A3 model, the original formal train
and locked-test files, and the separately validated portable training launcher.
The validated inference code is unchanged.

**Not bundled:** GFENet third-party upstream Python source; GFENet, U-head,
RETFound, and A3 checkpoints. The original RESMap wrapper and the separately
implemented U-head are preserved, but cannot generate RESMaps until GFENet
source is installed locally by the user.

### Real EyeQ examples (RESTRICTED REVIEW ONLY)

This REVIEW archive deliberately retains **10 real RGB + FOV-mask + uint16 RESMap trios**
and `examples/expected_predictions.csv` exactly as validated. **Their redistribution
rights have not been established. This archive MUST NOT be published to a public
GitHub repository, release page, or shared publicly until the rights are cleared.**

## A3 inference with PRECOMPUTED uint16 RESMaps

From this extracted directory (once you have the RETFound and A3 checkpoints):

```bash
python inference.py \
  --release-dir . \
  --image-dir examples/images \
  --resmap-dir examples/resmaps \
  --retfound-checkpoint /path/to/RETFound_mae_natureCFP.pth \
  --a3-checkpoint /path/to/best_srcc.pt \
  --a3-device cuda \
  --output-dir ./my_predictions
```

Each RESMap must be **16-bit single-channel PNG**, saved as
`clip(round(U * 65535), 0, 65535)`. Do not use 8-bit visualizations.

## OPTIONAL GFENet setup (from user's separately obtained checkout)

Upstream project: https://github.com/liamheng/Annotation-free-Fundus-Image-Enhancement

After obtaining lawful access to GFENet source, locally install the 10 model
files from your checkout WITHOUT uploading them to this public repository:

```bash
python restore_gfenet_locally.py --gfenet-source /path/to/Annotation-free-Fundus-Image-Enhancement
```

Then call `inference.py` with `--image-dir`, `--mask-dir`,
`--gfenet-checkpoint`, `--uhead-checkpoint`, `--retfound-checkpoint`,
`--a3-checkpoint`, `--generator-gpu -1`, `--a3-device cuda`, and `--output-dir`.
The original generator needs matching FOV masks, not RGB alone.

## Portable A3 training

`train.py` is the configurable entry; `tools/train_a3_portable.py` is the
path-adapted copy. Consult `PORTABLE_TRAIN_README.md` and set your own
`configs/portable_a3.yaml`. The original historical train/test source is
retained byte-identically, including historical server-path defaults. Do
not use the historical entry unchanged on a new machine.

This archive does not bundle training annotations or full patient-level data.
The trainer has passed **one** real Adam + EMA update, not a full training
reproduction. Environment setup is not verified on a separate fresh machine.

## Weights & third-party rights

No checkpoint is provided or mirrored. Users obtain the RETFound checkpoint
from https://huggingface.co/YukunZhou/RETFound_mae_natureCFP subject to its
access terms (CC BY-NC 4.0). GFENet upstream source and checkpoint have not
been cleared for redistribution. See `THIRD_PARTY_RIGHTS.md` and
`THIRD_PARTY_NOTICES.md`. The licenses of individual imported or adapted
implementations still require a file-level provenance check before public upload.

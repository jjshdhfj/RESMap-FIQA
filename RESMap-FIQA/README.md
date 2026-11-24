# RESMap-FIQA — Restricted 6,134-Image Dataset

## Scope

This document describes the **separately stored, access-restricted 6,134-image dataset** used for RESMap-FIQA research. It is **not** the GitHub source-code release and is **not** the 10-pair inference-example collection. Do not merge these collections or place this dataset under the public repository's `examples/` directory.

The photographs are drawn from EyeQ, which derives its imagery from EyePACS. The presence of this README or encryption does **not** grant permission to redistribute the images or related derivatives. Share this archive only where the applicable data terms and permissions allow it.

## Standalone archive layout

After authorized decryption and extraction, the expected structure is:

```text
RESMap-FIQA-6134/
├── README.md
└── data/
    ├── 13_left.jpeg
    ├── 13_right.jpeg
    ├── ...                         # 6,134 RGB fundus images
    ├── 原始.csv                    # Eight-rater records and aggregated quality scores
    └── final_real_score.csv       # Final four-attribute quality scores
```

All 6,134 photographs and both CSV files are placed **inside `data/`**. Keep original image filenames unchanged. The image names must match the `图片文件名` column of each CSV. This archive must not contain the separate 10 inference examples or any model checkpoints.

## Annotation files

The two supplied CSVs were checked for their basic structure:

| File | Image records | Columns | Encoding | Contents |
| --- | ---: | ---: | --- | --- |
| `data/原始.csv` | 6,134 | 41 | UTF-8 with BOM (`utf-8-sig`) | Image filename and, for each of four quality attributes, eight rater scores, their mean, and a final score |
| `data/final_real_score.csv` | 6,134 | 5 | GB18030 (`gb18030`) | Image filename and final scores for the four attributes |

The five column names in `final_real_score.csv` are exactly:

```text
图片文件名,清晰度,曝光,伪影,关键区域完整度
```

The labels use increments of 0.5. The observed recorded ranges are 0–5 for clarity, exposure and completeness, and 0–4 for artifact-related quality. The artifact column is a **quality score**, not a lesion annotation. These attributes are not the original three-class EyeQ labels.

If preparing the dataset from the files supplied for this project, rename `原始(3).csv` to `原始.csv` and `final_real_score(1).csv` to `final_real_score.csv`. Renaming must not change their contents.

## Validate the extracted dataset

Run from inside the extracted `RESMap-FIQA-6134/` directory:

```bash
python - <<'PY'
import csv
from pathlib import Path

root = Path('data')

def read_names(filename, encoding):
    with (root / filename).open(encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        assert '图片文件名' in (reader.fieldnames or []), filename
        names = [row['图片文件名'] for row in reader]
    assert len(names) == 6134, (filename, len(names))
    assert len(set(names)) == 6134, f'Duplicate filenames in {filename}'
    return names

raw = read_names('原始.csv', 'utf-8-sig')
final = read_names('final_real_score.csv', 'gb18030')
assert set(raw) == set(final), 'The two CSVs contain different image sets'
missing = [name for name in final if not (root / name).is_file()]
assert not missing, f'Missing image files: {missing[:10]} (total={len(missing)})'
print('[PASS] 6,134 CSV entries, unique filenames, and corresponding image files')
PY
```

This check establishes only basic file presence and naming consistency; it does not validate image licensing, patient-level split membership, or annotation quality.

## Experimental split and preprocessing

The original RESMap-FIQA experiments used a fixed patient-level split of **4,414 training / 493 validation / 1,227 test images**. The two label tables alone do **not** include the locked split assignments. Use the separately maintained official split manifests for reproducing the reported evaluation; do not randomly re-split this archive and call the metrics an exact reproduction.

The A3 model requires an RGB image paired with its RESMap. RESMaps are generated separately using the designated GFENet and U-head checkpoints and an image-matched field-of-view mask. The A3 input map is single-channel **uint16 PNG** with fixed encoding `clip(round(U × 65535), 0, 65535)`; it is decoded by dividing by 65535. Precomputed RESMaps, FOV masks, code and checkpoints are **not included** in this restricted RGB/labels archive unless separately listed in an authorized distribution manifest.

## Access and encrypted packaging

If you are authorized to store or transfer this dataset, protect the separate directory with a password-encrypted **7z** archive, using AES-256 and encrypted filenames. Example on Ubuntu, from the parent directory:

```bash
7z a -t7z -mhe=on -mx=1 -p RESMap-FIQA-6134-RESTRICTED.7z RESMap-FIQA-6134/
7z t RESMap-FIQA-6134-RESTRICTED.7z
```

`-p` prompts for a password; do not place the password in shell command history or in the repository. `-mhe=on` encrypts filenames. Keep the password separate from the encrypted archive. Encryption controls access but does **not** create a redistribution right. Do not upload this archive to a public GitHub repository or public Release unless the necessary data-sharing permissions have been established.

## Provenance

- EyeQ repository: https://github.com/HzFu/EyeQ
- EyePACS data access: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data
- Fu et al., *Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces*, MICCAI 2019.

The eight-rater annotations and aggregate quality scores are project-specific annotations; their presence here does not establish an independent right to redistribute the source photographs.

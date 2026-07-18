# Dataset Provenance

## Dataset

- Dataset name: 30VNFoods
- Kaggle dataset slug used by this pipeline: `quandang/vietnamese-foods`
- Original task: Vietnamese food image classification
- Original class count: 30 food classes
- Selected class count: 15 food classes

## Selected Classes

```text
pho
banh_mi
com_tam
bun_bo_hue
goi_cuon
banh_xeo
mi_quang
xoi_xeo
chao_long
bun_thit_nuong
bun_rieu
hu_tieu
banh_cuon
banh_trang_nuong
cao_lau
```

## Source And License Notes

Users must download the dataset themselves using Kaggle credentials:

```bash
python3 ai_training/scripts/download_dataset.py \
  --output ai_training/data/source
```

Images, archives, processed manifests, and model artifacts are not committed to
this repository. Keep the dataset license and Kaggle terms in mind before
redistributing any data or trained artifact.

The dataset may contain duplicates, label noise, corrupted files, unusual image
sizes, or classes that look visually similar. The pipeline reports these issues
but does not silently rewrite labels or claim the dataset is fully cleaned.

Every training or evaluation result should record the dataset manifest hash from
`ai_training/data/processed/dataset_manifest.json`.

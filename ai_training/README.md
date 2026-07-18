# NutriSnap AI Training

Pipeline này huấn luyện bộ phân loại ảnh món ăn Việt Nam cho 15 lớp từ 30VNFoods bằng TensorFlow/Keras MobileNetV2, sau đó chuyển sang TensorFlow Lite để chuẩn bị tích hợp Android.

## Cài Đặt

Windows:

```bash
cd ai_training
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux:

```bash
cd ai_training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Dataset không được commit. Xem thêm `DATASET.md` về provenance và license.

Tải bằng Kaggle sau khi cấu hình credential:

```bash
python3 ai_training/scripts/download_dataset.py \
  --output ai_training/data/source
```

Hoặc đặt dataset đã giải nén vào `ai_training/data/source`. Pipeline hỗ trợ folder
class trực tiếp hoặc layout lồng nhiều cấp. Ví dụ:

```text
ai_training/data/source/Phở/
ai_training/data/source/Bánh mì/
ai_training/data/source/Bún bò Huế/
...
```

Thứ tự labels nằm trong `metadata/food_labels.txt`, `metadata/class_names.json`, và `configs/mobilenetv2_config.yaml`. Không đổi thứ tự này sau khi đã train model.

## Kiểm Tra Dữ Liệu

Chạy từ root repository:

```bash
python ai_training/scripts/inspect_dataset.py \
  --config ai_training/configs/mobilenetv2_config.yaml
```

Output nằm trong `ai_training/outputs/reports/` và `ai_training/outputs/plots/`. Script chỉ tạo báo cáo, không xóa hoặc sửa file dataset.

## Chia Dữ Liệu

```bash
python ai_training/scripts/prepare_dataset.py \
  --config ai_training/configs/mobilenetv2_config.yaml
```

Output:

```text
ai_training/data/processed/train.csv
ai_training/data/processed/validation.csv
ai_training/data/processed/test.csv
ai_training/outputs/reports/split_summary.csv
```

Ảnh trùng SHA-256 được giữ trong cùng một split để giảm rủi ro data leakage.

## Huấn Luyện

Experiment protocol hiện tại:

```text
deduplicated_stratified_80_10_10_v1
```

Đây là custom split của NutriSnap sau duplicate filtering, không phải official
30VNFoods benchmark split. Phase AI-5/6 chỉ dùng train và validation; test set
được giữ nguyên cho Phase AI-7.

Kiểm tra GPU:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
nvidia-smi
```

Smoke test bắt buộc trước full training:

```bash
python3 ai_training/scripts/train.py \
  --config ai_training/configs/mobilenetv2_config.yaml \
  --smoke-test
```

```bash
python ai_training/scripts/train.py \
  --config ai_training/configs/mobilenetv2_config.yaml
```

Đổi batch size khi thiếu GPU memory:

```bash
python ai_training/scripts/train.py \
  --config ai_training/configs/mobilenetv2_config.yaml \
  --batch-size 16
```

Resume Stage 2 từ run có `checkpoints/best_head.keras`:

```bash
python ai_training/scripts/train.py \
  --config ai_training/configs/mobilenetv2_config.yaml \
  --resume-run ai_training/outputs/runs/YYYYMMDD_HHMMSS_mobilenetv2
```

Mỗi lần train tạo run riêng tại `ai_training/outputs/runs/YYYYMMDD_HHMMSS_mobilenetv2/`.
`ai_training/outputs/latest_run.txt` trỏ tới run mới nhất. Checkpoint, logs và
run artifacts không được commit.

Run directory chứa:

```text
config_resolved.yaml
run_metadata.json
dataset_manifest_snapshot.json
class_weights.json
model_summary.txt
logs/
checkpoints/
histories/
reports/
plots/
```

Xem TensorBoard:

```bash
tensorboard --logdir ai_training/outputs/runs
```

Stage 1 freeze MobileNetV2 backbone và train classification head. Stage 2 bắt
đầu từ `best_head.keras`, unfreeze các layer cuối theo config và giữ
BatchNormalization frozen. Model chưa được đánh giá trên test set trong Phase
AI-5/6.

## Đánh Giá

```bash
python ai_training/scripts/evaluate.py \
  --config ai_training/configs/mobilenetv2_config.yaml \
  --model ai_training/outputs/checkpoints/best_finetuned.keras
```

Output gồm metrics JSON, classification report, predictions, confusion matrix và error-analysis grids.

## Convert TFLite

```bash
python ai_training/scripts/convert_tflite.py \
  --config ai_training/configs/mobilenetv2_config.yaml \
  --model ai_training/outputs/checkpoints/best_finetuned.keras
```

Tạo Float32, Float16 và INT8 nếu có train manifest cho representative dataset. INT8 dùng train split, không dùng test set.

## Validate TFLite

```bash
python ai_training/scripts/validate_tflite.py \
  --config ai_training/configs/mobilenetv2_config.yaml
```

Script đọc dtype, shape, quantization parameters, chạy inference trên test split và xuất `ai_training/outputs/reports/tflite_comparison.csv`.

## Predict Một Ảnh

```bash
python ai_training/scripts/predict.py \
  --model ai_training/outputs/checkpoints/best_finetuned.keras \
  --image sample.jpg \
  --top-k 3
```

Preprocessing dùng chung: decode RGB, resize with pad về 224x224, scale về miền MobileNetV2 `preprocess_input` là `[-1, 1]`. Với TFLite INT8, script validate dùng `scale`, `zero_point`, và `dtype` thực tế từ interpreter.

## Android Export

Chưa copy vào `app/src/main/assets`. Chỉ chuẩn bị package:

```bash
python ai_training/scripts/export_android_assets.py \
  --model ai_training/outputs/tflite/nutrisnap_food_classifier_float16.tflite \
  --output android_export
```

Sau khi có xác nhận, mới copy package cuối vào Android assets.

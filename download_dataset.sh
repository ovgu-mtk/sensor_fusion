#!/usr/bin/env bash
set -e

# ================== KONFIG ==================
ZIP_URL="https://cloud.ovgu.de/public.php/dav/files/i32f79eCJCHp9Kn/?accept=zip"
ZIP_FILE="dataset.zip"
TARGET_DIR="dataset"
TMP_DIR="_tmp_extract"
# ============================================

echo "==> Downloading dataset ZIP (resume enabled)..."

wget \
  --continue \
  --progress=dot:giga \
  "$ZIP_URL" \
  -O "$ZIP_FILE"

echo "==> Extracting ZIP to temporary directory..."

rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

unzip -q "$ZIP_FILE" -d "$TMP_DIR"

# Struktur:
# _tmp_extract/
# └── sensor_fusion_dataset/
#     └── dataset/
#         ├── test/
#         └── train_val_test/

SRC_DATASET_DIR=$(find "$TMP_DIR" -type d -path "*/dataset" | head -n 1)

if [ -z "$SRC_DATASET_DIR" ]; then
  echo "❌ ERROR: dataset directory not found in ZIP"
  exit 1
fi

echo "==> Syncing files (skip existing)..."

mkdir -p "$TARGET_DIR"

rsync -av \
  --ignore-existing \
  "$SRC_DATASET_DIR/" \
  "$TARGET_DIR/"

echo "==> Cleaning up temporary files..."
rm -rf "$TMP_DIR"
rm -f "$ZIP_FILE"

echo "✅ Dataset download & extraction complete."

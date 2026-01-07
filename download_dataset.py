#!/usr/bin/env python3
import os
import shutil
import zipfile
import requests
from tqdm import tqdm

# ================== CONFIG ==================
ZIP_URL = "https://cloud.ovgu.de/public.php/dav/files/i32f79eCJCHp9Kn/?accept=zip"
ZIP_FILE = "dataset.zip"
TARGET_DIR = "dataset"
TMP_DIR = "_tmp_extract"
CHUNK_SIZE = 1024 * 1024  # 1 MB
# ============================================


def download_zip(url, output_path):
    """Download ZIP with resume support."""
    headers = {}
    mode = "wb"
    downloaded = 0

    if os.path.exists(output_path):
        downloaded = os.path.getsize(output_path)
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"

    response = requests.get(url, stream=True, headers=headers)
    response.raise_for_status()

    total_size = int(response.headers.get("Content-Length", 0)) + downloaded

    with open(output_path, mode) as f, tqdm(
        total=total_size,
        initial=downloaded,
        unit="B",
        unit_scale=True,
        desc="Downloading dataset.zip",
    ) as pbar:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    """Extract ZIP to temporary directory."""
    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def find_dataset_dir(base_dir):
    """Find the wrapped 'dataset' directory."""
    for root, dirs, _ in os.walk(base_dir):
        if "dataset" in dirs:
            return os.path.join(root, "dataset")
    return None


def sync_directories(src, dst):
    """Copy files, skipping already existing ones."""
    for root, _, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        target_root = os.path.join(dst, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_root, file)
            if os.path.exists(dst_file):
                continue
            shutil.copy2(src_file, dst_file)


def main():
    print("==> Downloading dataset ZIP (resume enabled)...")
    download_zip(ZIP_URL, ZIP_FILE)

    print("==> Extracting ZIP to temporary directory...")
    extract_zip(ZIP_FILE, TMP_DIR)

    print("==> Locating dataset directory...")
    src_dataset_dir = find_dataset_dir(TMP_DIR)
    if src_dataset_dir is None:
        raise RuntimeError("❌ ERROR: dataset directory not found in ZIP")

    print(f"Found dataset directory: {src_dataset_dir}")

    print("==> Syncing files (skip existing)...")
    os.makedirs(TARGET_DIR, exist_ok=True)
    sync_directories(src_dataset_dir, TARGET_DIR)

    print("==> Cleaning up temporary files and ZIP...")
    shutil.rmtree(TMP_DIR)
    os.remove(ZIP_FILE)

    print("✅ Dataset download & extraction complete.")


if __name__ == "__main__":
    main()

import os
import cv2
import random
import shutil
from glob import glob

import numpy as np
from tqdm import tqdm
import albumentations as A

# ==========================================================
# CONFIG
# ==========================================================
PROJECT_DIR = r"D:\Vision OPENCV\Code\OPENCVExample\Image Processing"
RAW_DIR = os.path.join(PROJECT_DIR, "raw")          # raw/class_name/*.jpg
OUT_DIR = os.path.join(PROJECT_DIR, "data5000")

IMG_SIZE = 128

TARGET_TRAIN_PER_CLASS = 4500
TARGET_VAL_PER_CLASS   = 500

SEED_TRAIN_RATIO = 0.8
RANDOM_SEED = 1337

CLEAR_OUTPUT_FIRST = True
COPY_ORIGINALS_TOO = True
# ==========================================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(folder, e)))
    return sorted(files)

def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        path += ".jpg"
    ok, buf = cv2.imencode(".jpg", img)
    if ok:
        buf.tofile(path)
    return ok

def pad_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

def normalize_img(img):
    img = pad_to_square(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img


# ==========================================================
# SAFE AUGMENTATION (KHÔNG PHÁ NÉT CHỮ)
# ==========================================================
def build_transforms(img_size):

    train_tf = A.Compose([
        A.Affine(
            scale=(0.92, 1.08),
            translate_percent=(-0.03, 0.03),
            rotate=(-8, 8),
            shear=(-3, 3),
            p=0.9
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.12,
            contrast_limit=0.12,
            p=0.6
        ),

        A.HueSaturationValue(
            hue_shift_limit=4,
            sat_shift_limit=6,
            val_shift_limit=6,
            p=0.4
        ),

        A.GaussianBlur(blur_limit=3, p=0.15),

        A.Resize(img_size, img_size)
    ])

    val_tf = A.Compose([
        A.Affine(
            scale=(0.96, 1.04),
            rotate=(-4, 4),
            translate_percent=(-0.02, 0.02),
            p=0.6
        ),

        A.RandomBrightnessContrast(0.08, 0.08, p=0.4),

        A.Resize(img_size, img_size)
    ])

    return train_tf, val_tf


def save_set(class_name, seeds, out_dir, target, tfm, prefix):
    ensure_dir(out_dir)
rng = random.Random(RANDOM_SEED + hash(class_name) % 99999)

    written = 0

    if COPY_ORIGINALS_TOO:
        for p in seeds:
            img = imread_unicode(p)
            if img is None:
                continue
            img = normalize_img(img)
            if imwrite_unicode(os.path.join(out_dir, f"{prefix}_orig_{written:05d}.jpg"), img):
                written += 1
            if written >= target:
                return

    pbar = tqdm(total=target - written, desc=f"{class_name}/{prefix}", leave=False)

    while written < target:
        src = rng.choice(seeds)
        img = imread_unicode(src)
        if img is None:
            continue
        img = normalize_img(img)
        aug = tfm(image=img)["image"]
        if imwrite_unicode(os.path.join(out_dir, f"{prefix}_aug_{written:05d}.jpg"), aug):
            written += 1
            pbar.update(1)

    pbar.close()


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if CLEAR_OUTPUT_FIRST and os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    train_tf, val_tf = build_transforms(IMG_SIZE)

    train_root = os.path.join(OUT_DIR, "train")
    val_root   = os.path.join(OUT_DIR, "val")

    ensure_dir(train_root)
    ensure_dir(val_root)

    classes = sorted(os.listdir(RAW_DIR))
    print("Classes:", classes)

    for cls in classes:
        imgs = list_images(os.path.join(RAW_DIR, cls))
        if len(imgs) < 10:
            print(f"[WARN] {cls} chỉ có {len(imgs)} ảnh")

        random.shuffle(imgs)
        split = max(1, int(len(imgs) * SEED_TRAIN_RATIO))
        train_seeds = imgs[:split]
        val_seeds   = imgs[split:] or imgs[:1]

        save_set(cls, train_seeds,
                 os.path.join(train_root, cls),
                 TARGET_TRAIN_PER_CLASS,
                 train_tf, "train")

        save_set(cls, val_seeds,
                 os.path.join(val_root, cls),
                 TARGET_VAL_PER_CLASS,
                 val_tf, "val")

    print("DONE")
    print("Train:", train_root)
    print("Val:", val_root)


if __name__ == "__main__":
    main()

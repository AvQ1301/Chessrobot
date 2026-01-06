import os
import json
import tensorflow as tf
from tensorflow import keras

# =========================
# CONFIG (SỬA CHO ĐÚNG)
# =========================
DATASET_DIR = r"D:\Vision OPENCV\Code\OPENCVExample\Image Processing\data5000"
OUT_DIR     = r"D:\Vision OPENCV\Code\OPENCVExample\Image Processing\models"

IMG_SIZE = 128
BATCH = 32
SEED = 1337

EPOCHS_HEAD = 8
EPOCHS_FINE = 15
# =========================

train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "val")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_MODEL = os.path.join(OUT_DIR, "xiangqi_piece_model.h5")
OUT_CLASSES = os.path.join(OUT_DIR, "classes.txt")
OUT_CLASS_MAP = os.path.join(OUT_DIR, "class_to_idx.json")
OUT_LOG = os.path.join(OUT_DIR, "training_log.csv")

# =========================
# LOAD DATA
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Num classes:", num_classes)

# save classes
with open(OUT_CLASSES, "w", encoding="utf-8") as f:
    for c in class_names:
        f.write(c + "\n")

with open(OUT_CLASS_MAP, "w", encoding="utf-8") as f:
    json.dump({c:i for i,c in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# =========================
# AUGMENTATION (NHẸ KHI TRAIN)
# Lưu ý: bạn đã augment offline rồi, nên ở đây chỉ cần nhẹ
# =========================
data_aug = keras.Sequential([
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.05),
    keras.layers.RandomContrast(0.10),
], name="data_aug_light")

# =========================
# MODEL - MobileNetV2 transfer learning
# =========================
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_aug(inputs)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.25)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(OUT_MODEL, monitor="val_accuracy", save_best_only=True, verbose=1),
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1),
    keras.callbacks.CSVLogger(OUT_LOG)
]

print("\n=== TRAIN HEAD (freeze backbone) ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=callbacks)

print("\n=== FINE-TUNE (unfreeze last part) ===")
base.trainable = True

# chỉ fine-tune 30% layers cuối để ổn định
fine_tune_at = int(len(base.layers) * 0.7)
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=callbacks)

print("\nDONE ✅")
print("Saved model:", OUT_MODEL)
print("Saved classes:", OUT_CLASSES)
print("Saved log:", OUT_LOG)

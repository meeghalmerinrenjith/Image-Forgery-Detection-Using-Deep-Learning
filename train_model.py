import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout,
                                     BatchNormalization, Input, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence
import tensorflow as tf

BASE_DIR    = r"C:\Users\rengi\Desktop\im_forg"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ELA_DIR     = os.path.join(BASE_DIR, "dataset_ela")
TEMP_ELA    = os.path.join(BASE_DIR, "temp_ela.jpg")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

def convert_to_ela(image_path, quality=90):
    try:
        original = Image.open(image_path).convert("RGB")
        original.save(TEMP_ELA, "JPEG", quality=quality)
        compressed = Image.open(TEMP_ELA).convert("RGB")

        ela_image = ImageChops.difference(original, compressed)
        extrema   = ela_image.getextrema()
        max_diff  = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        scale     = 255.0 / max_diff
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        return ela_image
    except:
        return Image.open(image_path).convert("RGB")

def prepare_ela_dataset(input_dir, output_dir):
    print(f"\nConverting {input_dir} to ELA...")
    for category in ["authentic", "forged"]:
        src = os.path.join(input_dir, category)
        dst = os.path.join(output_dir, category)
        os.makedirs(dst, exist_ok=True)

        files = os.listdir(src)
        for i, f in enumerate(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in {'.jpg', '.jpeg', '.png'}:
                src_path = os.path.join(src, f)
                dst_path = os.path.join(dst, os.path.splitext(f)[0] + ".jpg")
                if not os.path.exists(dst_path):
                    ela_img = convert_to_ela(src_path)
                    ela_img.save(dst_path, "JPEG", quality=95)
            if i % 500 == 0:
                print(f"  [{category}] {i}/{len(files)} done...")
    print(f"  Done! ELA images saved to {output_dir}")

class DualStreamGenerator(Sequence):
    def __init__(self, orig_dir, ela_dir, batch_size, img_size, augment=False, shuffle=True):
        self.img_size   = img_size
        self.batch_size = batch_size
        self.augment    = augment
        self.shuffle    = shuffle

        self.samples = []
        for label, category in enumerate(["authentic", "forged"]):
            orig_cat = os.path.join(orig_dir, category)
            ela_cat  = os.path.join(ela_dir,  category)
            for f in os.listdir(orig_cat):
                ext = os.path.splitext(f)[1].lower()
                if ext in {'.jpg', '.jpeg', '.png'}:
                    orig_path = os.path.join(orig_cat, f)
                    ela_name  = os.path.splitext(f)[0] + ".jpg"
                    ela_path  = os.path.join(ela_cat, ela_name)
                    if os.path.exists(ela_path):
                        self.samples.append((orig_path, ela_path, label))

        self.classes = np.array([s[2] for s in self.samples])
        self.on_epoch_end()
        print(f"  Found {len(self.samples)} samples in {orig_dir}")

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        orig_batch = []
        ela_batch  = []
        labels     = []

        for i in batch_indices:
            orig_path, ela_path, label = self.samples[i]

            orig_img = Image.open(orig_path).convert("RGB").resize(self.img_size)
            ela_img  = Image.open(ela_path).convert("RGB").resize(self.img_size)

            orig_arr = np.array(orig_img) / 255.0
            ela_arr  = np.array(ela_img)  / 255.0

            if self.augment and np.random.rand() > 0.5:
                orig_arr = np.fliplr(orig_arr)
                ela_arr  = np.fliplr(ela_arr)
            if self.augment and np.random.rand() > 0.5:
                orig_arr = np.flipud(orig_arr)
                ela_arr  = np.flipud(ela_arr)

            orig_batch.append(orig_arr)
            ela_batch.append(ela_arr)
            labels.append(label)

        return [np.array(orig_batch), np.array(ela_batch)], np.array(labels, dtype=np.float32)

prepare_ela_dataset(os.path.join(DATASET_DIR, "train"), os.path.join(ELA_DIR, "train"))
prepare_ela_dataset(os.path.join(DATASET_DIR, "test"),  os.path.join(ELA_DIR, "test"))

print("\nLoading data generators...")
train_gen = DualStreamGenerator(
    os.path.join(DATASET_DIR, "train"),
    os.path.join(ELA_DIR,     "train"),
    batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=True, shuffle=True
)
test_gen = DualStreamGenerator(
    os.path.join(DATASET_DIR, "test"),
    os.path.join(ELA_DIR,     "test"),
    batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=False, shuffle=False
)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

def build_dual_stream_model(img_size):
    
    shared_base = MobileNetV2(weights="imagenet", include_top=False,
                               input_shape=(*img_size, 3))
    shared_base.trainable = False

    input_orig = Input(shape=(*img_size, 3), name="input_original")
    feat_orig  = shared_base(input_orig, training=False)
    feat_orig  = GlobalAveragePooling2D(name="gap_orig")(feat_orig)

    
    input_ela  = Input(shape=(*img_size, 3), name="input_ela")
    feat_ela   = shared_base(input_ela, training=False)
    feat_ela   = GlobalAveragePooling2D(name="gap_ela")(feat_ela)

    
    merged = Concatenate(name="merge")([feat_orig, feat_ela])
    x = BatchNormalization(name="bn_1")(merged)
    x = Dropout(0.4, name="drop_1")(x)
    x = Dense(256, activation="relu", name="dense_1")(x)
    x = BatchNormalization(name="bn_2")(x)
    x = Dropout(0.3, name="drop_2")(x)
    x = Dense(128, activation="relu", name="dense_2")(x)
    x = Dropout(0.2, name="drop_3")(x)
    output = Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=[input_orig, input_ela], outputs=output)
    return model, shared_base


model, shared_base = build_dual_stream_model(IMG_SIZE)
model.summary()

print("\n🔒 Phase 1: Training with frozen MobileNetV2...\n")

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(BASE_DIR, "best_model.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
    )
]

history1 = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=15,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase1
)


print("\n🔓 Phase 2: Fine-tuning last 30 layers...\n")

shared_base.trainable = True
for layer in shared_base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase2 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(
        os.path.join(BASE_DIR, "best_model_finetuned.h5"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1
    )
]

history2 = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=30,
    class_weight=class_weight_dict,
    callbacks=callbacks_phase2
)

model.save(os.path.join(BASE_DIR, "model_mobilenetv2_dualstream_final.h5"))

print("\n✅ Training complete!")
print(f"📁 Best phase 1 model    : {BASE_DIR}\\best_model.h5")
print(f"📁 Best fine-tuned model : {BASE_DIR}\\best_model_finetuned.h5")
print(f"📁 Final model           : {BASE_DIR}\\model_mobilenetv2_dualstream_final.h5")
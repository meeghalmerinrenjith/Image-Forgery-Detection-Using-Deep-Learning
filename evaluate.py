import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from PIL import Image
import os

BASE_DIR   = r"C:\Users\rengi\Desktop\im_forg"
MODEL_PATH = os.path.join(BASE_DIR, "best_model_finetuned.h5")
TEST_ORIG  = os.path.join(BASE_DIR, "dataset", "test")
TEST_ELA   = os.path.join(BASE_DIR, "dataset_ela", "test")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

class DualStreamGenerator(Sequence):
    def __init__(self, orig_dir, ela_dir, batch_size, img_size):
        self.img_size   = img_size
        self.batch_size = batch_size
        self.samples    = []

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
        print(f"  Found {len(self.samples)} test samples")

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]
        orig_batch, ela_batch, labels = [], [], []

        for orig_path, ela_path, label in batch:
            orig_arr = np.array(Image.open(orig_path).convert("RGB").resize(self.img_size)) / 255.0
            ela_arr  = np.array(Image.open(ela_path).convert("RGB").resize(self.img_size))  / 255.0
            orig_batch.append(orig_arr)
            ela_batch.append(ela_arr)
            labels.append(label)

        return [np.array(orig_batch), np.array(ela_batch)], np.array(labels)

print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading test data...")
test_gen = DualStreamGenerator(TEST_ORIG, TEST_ELA, BATCH_SIZE, IMAGE_SIZE)

print("Running predictions...")
y_pred_probs = model.predict(test_gen)
y_pred       = (y_pred_probs > 0.5).astype(int).flatten()
y_true       = test_gen.classes

class_names = ["authentic", "forged"]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
plt.ylabel("Actual", fontsize=13)
plt.xlabel("Predicted", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("✅ Confusion matrix saved as confusion_matrix.png")

print("\n" + "=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_true, y_pred, target_names=class_names))

accuracy = np.mean(y_pred == y_true) * 100
print(f"Overall Accuracy: {accuracy:.2f}%")
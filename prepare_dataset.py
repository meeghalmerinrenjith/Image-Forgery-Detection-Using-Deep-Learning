import os
import shutil
import random

CASIA1_AU = r"C:\Users\rengi\Downloads\CASIA\casia\CASIA1\Au"
CASIA1_SP = r"C:\Users\rengi\Downloads\CASIA\casia\CASIA1\Sp"   # forged (spliced)

CASIA2_AU = r"C:\Users\rengi\Downloads\CASIA\casia\CASIA2\Au"
CASIA2_TP = r"C:\Users\rengi\Downloads\CASIA\casia\CASIA2\Tp"   # forged (tampered)

TRAIN_DIR = r"C:\Users\rengi\Desktop\im_forg\dataset\train"
TEST_DIR  = r"C:\Users\rengi\Desktop\im_forg\dataset\test"

SPLIT_RATIO      = 0.8
RANDOM_SEED      = 42
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def get_valid_images(folder):
    """Return list of (full_path, filename) for valid images in a folder."""
    if not os.path.exists(folder):
        print(f"  ⚠️  Folder not found, skipping: {folder}")
        return []
    result = []
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in VALID_EXTENSIONS:
            result.append((os.path.join(folder, f), f))
    skipped = len(os.listdir(folder)) - len(result)
    if skipped:
        print(f"  ⚠️  Skipped {skipped} non-image files in {os.path.basename(folder)}")
    return result


def make_unique_filename(filename, existing_names):
    """Add suffix to avoid filename collisions when merging datasets."""
    if filename not in existing_names:
        return filename
    name, ext = os.path.splitext(filename)
    counter = 1
    while f"{name}_{counter}{ext}" in existing_names:
        counter += 1
    return f"{name}_{counter}{ext}"


def split_and_copy(all_images, category):
    """Shuffle, split 80/20, and copy images to train/test folders."""
    random.shuffle(all_images)
    split_idx   = int(len(all_images) * SPLIT_RATIO)
    train_files = all_images[:split_idx]
    test_files  = all_images[split_idx:]

    train_cat_dir = os.path.join(TRAIN_DIR, category)
    test_cat_dir  = os.path.join(TEST_DIR,  category)
    os.makedirs(train_cat_dir, exist_ok=True)
    os.makedirs(test_cat_dir,  exist_ok=True)

    existing_train = set()
    existing_test  = set()

    for src_path, fname in train_files:
        unique_fname = make_unique_filename(fname, existing_train)
        existing_train.add(unique_fname)
        shutil.copy(src_path, os.path.join(train_cat_dir, unique_fname))

    for src_path, fname in test_files:
        unique_fname = make_unique_filename(fname, existing_test)
        existing_test.add(unique_fname)
        shutil.copy(src_path, os.path.join(test_cat_dir, unique_fname))

    print(f"  [{category}] Total: {len(all_images)} | Train: {len(train_files)} | Test: {len(test_files)}")

random.seed(RANDOM_SEED)

print("=" * 55)
print("  Dataset Preparation — CASIA1 + CASIA2 Merge")
print("=" * 55)

print("\n📂 Collecting AUTHENTIC images...")
casia1_au = get_valid_images(CASIA1_AU)
casia2_au = get_valid_images(CASIA2_AU)
authentic  = casia1_au + casia2_au
print(f"  CASIA1 Au : {len(casia1_au)} images")
print(f"  CASIA2 Au : {len(casia2_au)} images")
print(f"  Total     : {len(authentic)} images")

print("\n📂 Collecting FORGED images...")
casia1_sp = get_valid_images(CASIA1_SP)
casia2_tp = get_valid_images(CASIA2_TP)
forged     = casia1_sp + casia2_tp
print(f"  CASIA1 Sp : {len(casia1_sp)} images")
print(f"  CASIA2 Tp : {len(casia2_tp)} images")
print(f"  Total     : {len(forged)} images")

print(f"\n📊 Grand Total : {len(authentic) + len(forged)} images")
print(f"   Split       : {int(SPLIT_RATIO*100)}% train / {int((1-SPLIT_RATIO)*100)}% test\n")

print("📁 Copying files...")
split_and_copy(authentic, "authentic")
split_and_copy(forged,    "forged")

print("\n✅ Done! Folder structure created:")
print(f"  {TRAIN_DIR}\\authentic\\")
print(f"  {TRAIN_DIR}\\forged\\")
print(f"  {TEST_DIR}\\authentic\\")
print(f"  {TEST_DIR}\\forged\\")
"""
create_mini_dataset.py
======================

Build a SMALL development copy of the PickSense / OpenLORIS-Object "occlusion"
dataset that mirrors the size and folder layout of the Learn PyTorch
`pizza_steak_sushi` dataset (a few hundred images).

Why?
----
The full OpenLORIS dataset is very large. For quickly building and debugging a
CNN / Vision Transformer pipeline it is much faster to work with a tiny,
balanced subset first. This script creates that subset.

What it produces
----------------
    data/picksense_mini/
    ├── train/
    │   ├── clear/                (up to 1,000 images)
    │   ├── partially_occluded/   (up to 1,000 images)
    │   └── heavily_occluded/     (up to 1,000 images)
    └── test/
        ├── clear/                (up to 200 images)
        ├── partially_occluded/   (up to 200 images)
        └── heavily_occluded/     (up to 200 images)

    => Up to 3,000 train + 600 test = 3,600 images total.

Important guarantees
--------------------
* Your ORIGINAL dataset is never deleted, moved, renamed, or modified.
  This script only READS the originals and COPIES a few of them.
* The SAME image can never appear in both train and test.
* Sampling is reproducible because we use `random.seed(42)`.

How OpenLORIS "occlusion" is organised (and how we map it to classes)
---------------------------------------------------------------------
The OpenLORIS-Object "occlusion" condition looks like this:

    occlusion/
    ├── train/
    │   ├── task1/<object>/<frame>.jpg
    │   ├── task2/...
    │   └── ...   task9
    └── test/
        ├── task1/<object>/<frame>.jpg
        └── ...   task9

Following the PickSense README, the 9 tasks map to 3 occlusion classes:

    clear              -> task1, task2, task3   (~0%  occlusion)
    partially_occluded -> task4, task5, task6   (~25% occlusion)
    heavily_occluded   -> task7, task8, task9   (~50% occlusion)

Because OpenLORIS already ships separate `train/` and `test/` folders, we draw
train images only from `occlusion/train` and test images only from
`occlusion/test`. Those are physically different files, so train and test can
never overlap.

Running it
----------
    # From the repository root (locally, or in Colab/Kaggle after cd-ing in):
    python src/create_mini_dataset.py

    # Optional: point it at a specific occlusion folder if auto-detect fails:
    python src/create_mini_dataset.py /kaggle/input/openlorisobject/openloris/occlusion

Only the Python standard library is used, so no extra packages are required.
"""

import os
import sys
import random
import shutil


# ---------------------------------------------------------------------------
# 1. CONFIGURATION  (edit these if you need to)
# ---------------------------------------------------------------------------

# Make the random sampling reproducible: same images chosen every run.
RANDOM_SEED = 42

# Where the small dataset will be written (relative to your current folder).
DEST_DIR = os.path.join("data", "picksense_mini")

# How many images we want per class, for each split.
TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 200

# Image file types we accept (compared in lower-case, so .JPG also works).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# Fixed class order so the printed summary always looks the same.
CLASS_ORDER = ["clear", "partially_occluded", "heavily_occluded"]

# Which OpenLORIS "task" folders belong to each occlusion class.
CLASS_TO_TASKS = {
    "clear": ["task1", "task2", "task3"],
    "partially_occluded": ["task4", "task5", "task6"],
    "heavily_occluded": ["task7", "task8", "task9"],
}

# Set this to a path string to skip auto-detection, e.g.
#   SOURCE_OCCLUSION_DIR = "/kaggle/input/openlorisobject/openloris/occlusion"
# Leave as None to let the script search for the "occlusion" folder itself.
SOURCE_OCCLUSION_DIR = None

# Places the script will look for an "occlusion" folder when auto-detecting.
# Covers the common Kaggle, Google Colab, and local layouts.
CANDIDATE_SOURCE_ROOTS = [
    "/kaggle/input/openlorisobject",
    "/content/picksense/data/raw/openloris",
    os.path.join("data", "raw", "openloris"),
    os.path.join("data", "raw"),
    "data",
    ".",
]

# Heavy or irrelevant folders we never want to walk into while searching.
SKIP_DIR_NAMES = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", ".cache", ".ipynb_checkpoints",
}


# ---------------------------------------------------------------------------
# 2. SMALL HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def find_occlusion_dir(candidate_roots):
    """Search the candidate roots and return the first 'occlusion' folder found.

    Returns the folder path as a string, or None if nothing was found.
    """
    for root in candidate_roots:
        if not os.path.isdir(root):
            continue

        # The root itself might already be the occlusion folder.
        if os.path.basename(os.path.normpath(root)).lower() == "occlusion":
            return root

        # Otherwise walk downwards looking for a folder literally named
        # "occlusion" (OpenLORIS also has illumination/pixel/etc., which we
        # intentionally skip).
        for dirpath, dirnames, _filenames in os.walk(root):
            # Prune folders we never care about (keeps the search fast).
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for d in dirnames:
                if d.lower() == "occlusion":
                    return os.path.join(dirpath, d)
    return None


def detect_split_dirs(occlusion_dir):
    """Return {'train': path, 'test': path} if both split folders exist.

    OpenLORIS ships pre-made train/ and test/ folders. If they are present we
    use them (this is what keeps train and test from ever overlapping).
    Returns None if the split folders are not there.
    """
    train_dir = os.path.join(occlusion_dir, "train")
    test_dir = os.path.join(occlusion_dir, "test")
    if os.path.isdir(train_dir) and os.path.isdir(test_dir):
        return {"train": train_dir, "test": test_dir}
    return None


def list_images_for_tasks(base_dir, tasks):
    """Collect every image path under the given task folders inside base_dir.

    We walk each task folder recursively (task -> object -> frames) and keep
    only files whose extension is in IMAGE_EXTENSIONS. The result is sorted so
    that the random sampling is identical on every machine.
    """
    files = []
    for task in tasks:
        task_dir = os.path.join(base_dir, task)
        if not os.path.isdir(task_dir):
            # A missing task just means fewer images; we do not crash.
            continue
        for dirpath, dirnames, filenames in os.walk(task_dir):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
            for filename in filenames:
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    files.append(os.path.join(dirpath, filename))
    files.sort()  # deterministic order -> reproducible sampling
    return files


def make_unique_name(src_path, base_dir):
    """Build a collision-free destination filename from the source path.

    Many OpenLORIS objects reuse names like 'frame0001.jpg', so copying by the
    original filename would overwrite files. We turn the path relative to the
    split folder (e.g. 'task1/cup_01/frame0093.jpg') into a flat, unique name
    ('task1_cup_01_frame0093.jpg').
    """
    rel = os.path.relpath(src_path, base_dir)
    return rel.replace(os.sep, "_").replace("/", "_")


def reset_dest_dir(dest_dir):
    """Delete a previous mini dataset (if any) so counts are always exact.

    A safety check makes sure we can ONLY ever delete a folder named
    'picksense_mini' -- never your source data.
    """
    if os.path.basename(os.path.normpath(dest_dir)) != "picksense_mini":
        raise ValueError(
            "Refusing to delete '%s': DEST_DIR must end with 'picksense_mini'."
            % dest_dir
        )
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)


def copy_files(selected_paths, base_dir, dest_class_dir):
    """Copy each selected source image into the destination class folder."""
    os.makedirs(dest_class_dir, exist_ok=True)
    for src in selected_paths:
        dest_name = make_unique_name(src, base_dir)
        # copy2 preserves timestamps; it reads the original and writes a copy,
        # so the original file is never changed.
        shutil.copy2(src, os.path.join(dest_class_dir, dest_name))


def count_images_in(directory):
    """Count image files directly inside a folder (used for the final summary)."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1 for name in os.listdir(directory)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def directory_size_bytes(path):
    """Add up the size of every file under a folder, in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.isfile(file_path):
                total += os.path.getsize(file_path)
    return total


def human_readable_size(num_bytes):
    """Turn a byte count into a friendly string like '12.3 MB'."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return "%.1f %s" % (size, unit)
        size /= 1024
    return "%.1f TB" % size


# ---------------------------------------------------------------------------
# 3. MAIN LOGIC
# ---------------------------------------------------------------------------

def main():
    # (a) Work out where the source "occlusion" folder is.
    #     Priority: command-line argument > SOURCE_OCCLUSION_DIR > auto-detect.
    if len(sys.argv) > 1:
        occlusion_dir = sys.argv[1]
    elif SOURCE_OCCLUSION_DIR is not None:
        occlusion_dir = SOURCE_OCCLUSION_DIR
    else:
        occlusion_dir = find_occlusion_dir(CANDIDATE_SOURCE_ROOTS)

    # If we still could not find it, explain how to fix it and stop (no crash).
    if not occlusion_dir or not os.path.isdir(occlusion_dir):
        print("Could not locate the OpenLORIS 'occlusion' folder.")
        print("Searched these locations:")
        for root in CANDIDATE_SOURCE_ROOTS:
            print("  - %s" % os.path.abspath(root))
        print()
        print("Fix it in one of these ways:")
        print("  1) Pass the path directly:")
        print("       python src/create_mini_dataset.py /path/to/openloris/occlusion")
        print("  2) Or edit SOURCE_OCCLUSION_DIR near the top of this script.")
        return

    print("Using source occlusion folder: %s" % os.path.abspath(occlusion_dir))

    # (b) Decide how to draw train vs test images.
    #     Preferred: OpenLORIS already has train/ and test/ subfolders.
    #     Fallback: no split folders -> we split one pool ourselves.
    split_dirs = detect_split_dirs(occlusion_dir)
    if split_dirs is not None:
        print("Found OpenLORIS train/ and test/ splits -> using them directly.\n")
        run_split_mode(split_dirs)
    else:
        print("No train/ and test/ splits found -> splitting a shared pool.\n")
        run_pooled_mode(occlusion_dir)


def run_split_mode(split_dirs):
    """Sample train from occlusion/train and test from occlusion/test.

    Because the two splits are separate files on disk, an image can never end
    up in both train and test.
    """
    needed_per_split = {"train": TRAIN_PER_CLASS, "test": TEST_PER_CLASS}

    # ---- STEP 1: gather all available images and CHECK counts BEFORE copying.
    available = {}  # (split, class) -> list of image paths
    print("Checking how many images are available per class...")
    for split, base_dir in split_dirs.items():
        for class_name in CLASS_ORDER:
            files = list_images_for_tasks(base_dir, CLASS_TO_TASKS[class_name])
            available[(split, class_name)] = files
            needed = needed_per_split[split]
            status = "OK" if len(files) >= needed else "NOT ENOUGH"
            print("  [%-5s] %-18s found %5d (need %d)  %s"
                  % (split, class_name, len(files), needed, status))
            if len(files) < needed:
                print("  WARNING: '%s' in '%s' has only %d images but %d are "
                      "needed. Copying all %d available."
                      % (class_name, split, len(files), needed, len(files)))
    print()

    # ---- STEP 2: reset the destination and copy the sampled images.
    reset_dest_dir(DEST_DIR)
    random.seed(RANDOM_SEED)  # seed once, right before we start sampling

    train_sources = []  # collected only to double-check train/test do not overlap
    test_sources = []

    for split, base_dir in split_dirs.items():
        needed = needed_per_split[split]
        for class_name in CLASS_ORDER:
            files = available[(split, class_name)]
            # Take a random sample, or everything we have if there are too few.
            if len(files) >= needed:
                selected = random.sample(files, needed)
            else:
                selected = list(files)

            dest_class_dir = os.path.join(DEST_DIR, split, class_name)
            copy_files(selected, base_dir, dest_class_dir)

            if split == "train":
                train_sources.extend(selected)
            else:
                test_sources.extend(selected)

    # ---- STEP 3: verify no image is shared between train and test.
    overlap = set(train_sources) & set(test_sources)
    if overlap:
        print("WARNING: %d image(s) appear in both train and test!" % len(overlap))
    else:
        print("Check passed: no image appears in both train and test.\n")

    print_summary()


def run_pooled_mode(occlusion_dir):
    """Fallback for sources that do NOT have train/ and test/ folders.

    For each class we build one pool of images, sample the requested total, then
    split it into train and test portions. Sampling without replacement
    guarantees train and test never share an image.
    """
    total_needed = TRAIN_PER_CLASS + TEST_PER_CLASS  # 100 per class

    # ---- STEP 1: gather and CHECK counts BEFORE copying.
    pools = {}  # class -> list of image paths
    print("Checking how many images are available per class...")
    for class_name in CLASS_ORDER:
        files = list_images_for_tasks(occlusion_dir, CLASS_TO_TASKS[class_name])
        pools[class_name] = files
        status = "OK" if len(files) >= total_needed else "NOT ENOUGH"
        print("  %-18s found %5d (need %d)  %s"
              % (class_name, len(files), total_needed, status))
        if len(files) < total_needed:
            print("  WARNING: '%s' has only %d images but %d are needed "
                  "(%d train + %d test). Copying as many as possible."
                  % (class_name, len(files), total_needed,
                     TRAIN_PER_CLASS, TEST_PER_CLASS))
    print()

    # ---- STEP 2: reset destination, then sample + copy.
    reset_dest_dir(DEST_DIR)
    random.seed(RANDOM_SEED)

    for class_name in CLASS_ORDER:
        files = pools[class_name]
        take = min(total_needed, len(files))
        chosen = random.sample(files, take)          # unique images, no repeats
        train_selected = chosen[:TRAIN_PER_CLASS]
        test_selected = chosen[TRAIN_PER_CLASS:]

        copy_files(train_selected, occlusion_dir,
                   os.path.join(DEST_DIR, "train", class_name))
        copy_files(test_selected, occlusion_dir,
                   os.path.join(DEST_DIR, "test", class_name))

    print("Check passed: train and test were sliced from the same non-repeating "
          "sample, so they cannot overlap.\n")
    print_summary()


def print_summary():
    """Print the per-class counts, totals, and the dataset size on disk.

    Counts are read back from the files actually on disk, so the summary always
    reflects reality (useful when a class had too few images).
    """
    print("=" * 40)
    print("PickSense Mini Dataset Created")
    print("=" * 40)

    grand_total = 0
    for split in ["train", "test"]:
        print()
        print("%s:" % split.capitalize())
        split_total = 0
        for class_name in CLASS_ORDER:
            count = count_images_in(os.path.join(DEST_DIR, split, class_name))
            print("%s: %d" % (class_name, count))
            split_total += count
        print("Total %s: %d" % (split, split_total))
        grand_total += split_total

    print()
    print("Total dataset: %d images" % grand_total)

    # Report where the data lives and how big it is on disk.
    size = directory_size_bytes(DEST_DIR)
    print()
    print("Dataset location: %s" % os.path.abspath(DEST_DIR))
    print("Dataset size on disk: %s" % human_readable_size(size))


if __name__ == "__main__":
    main()

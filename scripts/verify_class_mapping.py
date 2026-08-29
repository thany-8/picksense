#!/usr/bin/env python3
"""Verify the PickSense class-to-index mapping is consistent across:

  * the dataset (torchvision.datasets.ImageFolder.class_to_idx),
  * the trained checkpoint (a bare state_dict with no label metadata),
  * the backend inference code (app.backend.model_utils.CLASS_NAMES).

It reuses the *real* backend predictor, so this check cannot drift from what
the API actually serves.

Usage
-----
    python scripts/verify_class_mapping.py [--data DIR] [--model PATH]

``DIR`` must contain ``test/<class>/<images>`` (the picksense_mini test split).
Defaults:
    --data   PICKSENSE_DATA_DIR  or  /content/drive/MyDrive/PickSense/data/picksense_mini
    --model  PICKSENSE_MODEL_PATH or  <repo>/models/pretrained_vit_picksense.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torchvision import datasets  # noqa: E402

from app.backend.model_utils import CLASS_NAMES, PickSensePredictor  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=os.getenv(
            "PICKSENSE_DATA_DIR",
            "/content/drive/MyDrive/PickSense/data/picksense_mini",
        ),
        help="picksense_mini directory containing a test/ split.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "PICKSENSE_MODEL_PATH",
            str(REPO_ROOT / "models" / "pretrained_vit_picksense.pth"),
        ),
        help="Path to pretrained_vit_picksense.pth.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend_map = {name: index for index, name in enumerate(CLASS_NAMES)}

    print("Backend inference mapping (model output index -> class):")
    for index, name in enumerate(CLASS_NAMES):
        print(f"   {index} -> {name}")

    test_dir = Path(args.data) / "test"
    if not test_dir.is_dir():
        print(f"\n[!] Test split not found at: {test_dir}")
        print("    Run this where picksense_mini lives (e.g. Colab/Drive) or pass --data.")
        print("    The mapping above is what inference uses; compare it to")
        print("    ImageFolder(test_dir).class_to_idx on the training machine.")
        return 2

    test_dataset = datasets.ImageFolder(test_dir)
    dataset_map = test_dataset.class_to_idx
    idx_to_class = {index: name for name, index in dataset_map.items()}

    print("\nDataset ground-truth mapping (ImageFolder.class_to_idx):")
    print(f"   {dataset_map}")

    matches = dataset_map == backend_map
    print("\nDo training and inference mappings match?  ->  "
          f"{'YES ✅' if matches else 'NO ❌'}")
    if not matches:
        print(f"   dataset: {dataset_map}")
        print(f"   backend: {backend_map}")

    # One known image per class: file path | true label | true idx | pred idx | pred class
    predictor = PickSensePredictor(Path(args.model))
    print("\nfile path | true label | true idx | predicted idx | predicted class")
    print("-" * 96)

    seen: set[int] = set()
    for path, true_index in test_dataset.samples:
        if true_index in seen:
            continue
        seen.add(true_index)
        with open(path, "rb") as handle:
            prediction = predictor.predict(handle.read())["prediction"]
        predicted_index = CLASS_NAMES.index(prediction)
        print(
            f"{path} | {idx_to_class[true_index]} | {true_index} | "
            f"{predicted_index} | {prediction}"
        )
        if len(seen) == len(test_dataset.classes):
            break

    return 0 if matches else 1


if __name__ == "__main__":
    raise SystemExit(main())

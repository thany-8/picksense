#!/usr/bin/env python3
"""Evaluate the current PickSense checkpoint on a real-world validation set.

This is an EVALUATION-ONLY tool (no training, no UI changes). It reuses the real
backend predictor, so results match exactly what the API serves. Use it to capture
a baseline on phone photos, then re-run after augmentation / fine-tuning to compare.

Expected layout (folder names must match the training classes exactly):

    <data>/clear/*.jpg
    <data>/partially_occluded/*.jpg
    <data>/heavily_occluded/*.jpg

Usage
-----
    python scripts/evaluate_realworld.py [--data DIR] [--model PATH] [--show-errors]

Defaults:
    --data   PICKSENSE_REALWORLD_DIR or <repo>/data/realworld_val
    --model  PICKSENSE_MODEL_PATH    or <repo>/models/pretrained_vit_picksense.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from torchvision import datasets  # noqa: E402

from app.backend.model_utils import CLASS_NAMES, PickSensePredictor  # noqa: E402

# Same thresholds the UI uses to flag a prediction as low-confidence / out-of-distribution.
LOW_CONFIDENCE_PROB = 0.55
LOW_CONFIDENCE_MARGIN = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=os.getenv("PICKSENSE_REALWORLD_DIR", str(REPO_ROOT / "data" / "realworld_val")),
        help="Directory with clear/ partially_occluded/ heavily_occluded/ subfolders.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("PICKSENSE_MODEL_PATH", str(REPO_ROOT / "models" / "pretrained_vit_picksense.pth")),
        help="Path to pretrained_vit_picksense.pth.",
    )
    parser.add_argument("--show-errors", action="store_true", help="Print each misclassified image.")
    return parser.parse_args()


def is_low_confidence(probabilities: dict[str, float]) -> bool:
    ranked = sorted(probabilities.values(), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    return top < LOW_CONFIDENCE_PROB or (top - second) < LOW_CONFIDENCE_MARGIN


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data)

    print("Backend mapping (model output index -> class):")
    for index, name in enumerate(CLASS_NAMES):
        print(f"   {index} -> {name}")

    if not data_dir.is_dir():
        print(f"\n[!] Validation set not found at: {data_dir}")
        print("    Create it as <data>/{clear,partially_occluded,heavily_occluded}/*.jpg")
        print("    then re-run (or pass --data / set PICKSENSE_REALWORLD_DIR).")
        return 2

    dataset = datasets.ImageFolder(data_dir)
    if dataset.class_to_idx != {name: i for i, name in enumerate(CLASS_NAMES)}:
        print("\n[!] Folder class_to_idx does not match backend CLASS_NAMES:")
        print(f"    dataset: {dataset.class_to_idx}")
        print(f"    backend: {{name: i for i, name in enumerate(CLASS_NAMES)}}")
        print("    Fix the folder names before trusting the numbers below.")
        return 1

    predictor = PickSensePredictor(Path(args.model))
    n = len(dataset.samples)
    print(f"\nEvaluating {n} images from {data_dir} ...\n")

    num_classes = len(CLASS_NAMES)
    confusion = [[0] * num_classes for _ in range(num_classes)]
    per_class_total: Counter[int] = Counter()
    correct = 0
    confidence_sum = 0.0
    low_conf = 0
    errors: list[str] = []

    for path, true_index in dataset.samples:
        with open(path, "rb") as handle:
            out = predictor.predict(handle.read())
        pred_index = CLASS_NAMES.index(out["prediction"])
        confusion[true_index][pred_index] += 1
        per_class_total[true_index] += 1
        confidence_sum += out["probability"]
        if is_low_confidence(out["probabilities"]):
            low_conf += 1
        if pred_index == true_index:
            correct += 1
        elif args.show_errors:
            errors.append(
                f"  {path} | true={CLASS_NAMES[true_index]} | pred={CLASS_NAMES[pred_index]} "
                f"| conf={out['probability']:.1%}"
            )

    print("Per-class accuracy:")
    for index, name in enumerate(CLASS_NAMES):
        total = per_class_total[index]
        hits = confusion[index][index]
        pct = f"{hits / total:.1%}" if total else "n/a"
        print(f"   {name:<20} {hits}/{total} ({pct})")
    print(f"\nOverall accuracy: {correct}/{n} ({correct / n:.1%})" if n else "No images found.")

    short = [c.split("_")[0][:7] for c in CLASS_NAMES]
    print("\nConfusion matrix (rows = true, cols = predicted); order = CLASS_NAMES:")
    print("            " + "".join(f"{s:>9}" for s in short))
    for index, name in enumerate(CLASS_NAMES):
        print(f"   {short[index]:<8}" + "".join(f"{v:>9}" for v in confusion[index]))

    if n:
        print(f"\nMean top-1 confidence: {confidence_sum / n:.1%}")
        print(
            f"Low-confidence / OOD-flagged (top<{LOW_CONFIDENCE_PROB:.0%} or "
            f"margin<{LOW_CONFIDENCE_MARGIN:.0%}): {low_conf}/{n} ({low_conf / n:.1%})"
        )
    if args.show_errors and errors:
        print("\nMisclassified images:")
        print("\n".join(errors))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

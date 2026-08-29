#!/usr/bin/env python3
"""Fine-tune PickSense for better real-world (phone-photo) generalization.

What this does (and does NOT do):
  * Warm-starts from the existing checkpoint (ImageNet backbone + trained head).
  * Trains with real-world augmentation (see data_setup.build_transforms).
  * Unfreezes the last few transformer blocks + final LayerNorm + head, and
    fine-tunes them at a low learning rate. The rest of the backbone stays frozen.
  * Evaluates each epoch on the OpenLORIS test split (regression check) and, if
    provided, on a real-world validation set (the true target metric).
  * Saves the best checkpoint under a NEW name so the working model is never
    overwritten until you verify the result.

It does NOT change the class mapping, the input normalization, the backend, or
the UI. Run it where the dataset + a GPU live (e.g. Colab).

Example (Colab, after mounting Drive):
    python src/finetune.py \
        --data /content/drive/MyDrive/PickSense/data/picksense_mini \
        --realworld /content/drive/MyDrive/PickSense/data/realworld_val \
        --out /content/drive/MyDrive/PickSense/models/pretrained_vit_picksense_finetuned.pth \
        --epochs 8 --unfreeze-blocks 2
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets
from torchvision.models import ViT_B_16_Weights, vit_b_16

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from data_setup import build_transforms, create_percentage_dataloaders  # noqa: E402

REPO_ROOT = SRC_DIR.parent


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, checkpoint: Path, device: torch.device) -> nn.Module:
    if checkpoint and checkpoint.is_file():
        model = vit_b_16(weights=None)
        model.heads = nn.Linear(model.hidden_dim, num_classes)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"Warm-started from checkpoint: {checkpoint}")
    else:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Linear(model.hidden_dim, num_classes)
        print("Checkpoint not found -> started from ImageNet backbone + fresh head.")
    return model.to(device)


def collect_trainable(model: nn.Module, unfreeze_blocks: int) -> tuple[list, list]:
    for param in model.parameters():
        param.requires_grad = False

    backbone_params: list = []
    if unfreeze_blocks > 0:
        for block in model.encoder.layers[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
                backbone_params.append(param)
        for param in model.encoder.ln.parameters():
            param.requires_grad = True
            backbone_params.append(param)

    for param in model.heads.parameters():
        param.requires_grad = True
    head_params = list(model.heads.parameters())
    return backbone_params, head_params


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return running_loss / max(total, 1), correct / max(total, 1)


@torch.inference_mode()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / max(total, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="picksense_mini dir with train/ and test/.")
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "pretrained_vit_picksense.pth"),
                        help="Starting checkpoint to warm-start from.")
    parser.add_argument("--out", default=str(REPO_ROOT / "models" / "pretrained_vit_picksense_finetuned.pth"),
                        help="Where to save the best fine-tuned checkpoint (new name).")
    parser.add_argument("--realworld", default=None, help="Optional real-world val dir (per-class folders).")
    parser.add_argument("--realworld-train", default=None,
                        help="Optional dir of real-world TRAINING photos (per-class folders) to mix into "
                             "training. Keep these separate from --realworld to avoid leakage.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="Number of top encoder blocks to unfreeze.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--train-percentage", type=float, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seeds(args.seed)
    train_transform, eval_transform = build_transforms(augment=True)

    data_dir = Path(args.data)
    train_loader, test_loader, class_names = create_percentage_dataloaders(
        train_dir=data_dir / "train",
        test_dir=data_dir / "test",
        transform=eval_transform,          # deterministic for the test split
        batch_size=args.batch_size,
        train_percentage=args.train_percentage,
        seed=args.seed,
        num_workers=args.num_workers,
        train_transform=train_transform,   # augmentation for the train split only
    )
    print(f"Classes: {class_names}")

    if args.realworld_train:
        rw_train = datasets.ImageFolder(args.realworld_train, transform=train_transform)
        if rw_train.classes != class_names:
            raise SystemExit(
                f"real-world-train classes {rw_train.classes} != training {class_names}"
            )
        combined = ConcatDataset([train_loader.dataset, rw_train])
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(
            combined,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
        print(f"Mixed in {len(rw_train)} real-world training images "
              f"(train set now {len(combined)} images).")

    realworld_loader = None
    if args.realworld:
        rw = datasets.ImageFolder(args.realworld, transform=eval_transform)
        realworld_loader = DataLoader(rw, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)
        if rw.classes != class_names:
            print(f"[!] real-world classes {rw.classes} differ from training {class_names}")

    model = build_model(len(class_names), Path(args.model), device)
    set_seeds(args.seed)
    backbone_params, head_params = collect_trainable(model, args.unfreeze_blocks)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} "
          f"(unfrozen top blocks: {args.unfreeze_blocks})")

    groups = [{"params": head_params, "lr": args.lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_metric = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_acc = evaluate(model, test_loader, device)
        line = (f"Epoch {epoch:2d} | train_loss {train_loss:.4f} | train_acc {train_acc:.3f} "
                f"| openloris_test_acc {test_acc:.3f}")
        # Prefer the real-world metric for model selection when available.
        selection_metric = test_acc
        if realworld_loader is not None:
            rw_acc = evaluate(model, realworld_loader, device)
            line += f" | realworld_acc {rw_acc:.3f}"
            selection_metric = rw_acc
        print(line)

        if selection_metric > best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)

    metric_name = "realworld_acc" if realworld_loader is not None else "openloris_test_acc"
    print(f"\nBest {metric_name}={best_metric:.3f} at epoch {best_epoch}.")
    print(f"Saved best checkpoint to: {args.out}")
    print("Verify it, then (only if better) point the backend at it via "
          "PICKSENSE_MODEL_PATH or by replacing models/pretrained_vit_picksense.pth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

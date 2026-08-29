import math
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ViT_B_16_Weights


def build_transforms(augment: bool = True) -> tuple[Callable, Callable]:
	"""Return ``(train_transform, eval_transform)`` for the pretrained ViT.

	``eval_transform`` is the backbone's own preset (resize 256 -> center-crop 224
	-> ImageNet normalize) and MUST stay identical to what the backend uses at
	inference. When ``augment`` is True the training transform adds real-world
	robustness (object scale, flip, colour/lighting, viewpoint, mild blur) while
	ending with the *same* normalization, so nothing about the model input contract
	changes. Use ``eval_transform`` for validation/test so metrics stay comparable.
	"""
	weights = ViT_B_16_Weights.DEFAULT
	eval_transform = weights.transforms()
	mean, std = eval_transform.mean, eval_transform.std
	size = eval_transform.crop_size[0]

	if not augment:
		return eval_transform, eval_transform

	train_transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
			transforms.RandomRotation(15),
			transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
			transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
			transforms.ToTensor(),
			transforms.Normalize(mean, std),
		]
	)
	return train_transform, eval_transform


def _stratified_subset_indices(
	targets: list[int],
	percentage: float,
	seed: int,
) -> list[int]:
	"""Select a deterministic percentage from every class."""
	indices_by_class: dict[int, list[int]] = {}
	for index, target in enumerate(targets):
		indices_by_class.setdefault(target, []).append(index)

	selected_indices = []
	for target in sorted(indices_by_class):
		class_indices = indices_by_class[target]
		random.Random(seed + target).shuffle(class_indices)
		sample_count = max(1, math.ceil(len(class_indices) * percentage / 100))
		selected_indices.extend(class_indices[:sample_count])

	return selected_indices


def create_percentage_dataloaders(
	train_dir: str | Path,
	test_dir: str | Path,
	transform: Callable,
	batch_size: int,
	train_percentage: float = 100,
	seed: int = 42,
	num_workers: int = 0,
	train_transform: Callable | None = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
	"""Create loaders using a balanced percentage of training data and all test data.

	Increasing ``train_percentage`` with the same seed preserves previously selected
	samples, which makes progressive experiments directly comparable.

	``transform`` is applied to the test split (and to train when no separate
	``train_transform`` is given). Pass ``train_transform`` to augment the training
	split only, keeping the evaluation transform deterministic.
	"""
	if not 0 < train_percentage <= 100:
		raise ValueError("train_percentage must be greater than 0 and at most 100")
	if batch_size < 1:
		raise ValueError("batch_size must be at least 1")

	full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform or transform)
	test_dataset = datasets.ImageFolder(test_dir, transform=transform)

	selected_indices = _stratified_subset_indices(
		full_train_dataset.targets,
		percentage=train_percentage,
		seed=seed,
	)
	train_dataset = Subset(full_train_dataset, selected_indices)

	generator = torch.Generator().manual_seed(seed)
	loader_options = {
		"batch_size": batch_size,
		"num_workers": num_workers,
		"pin_memory": torch.cuda.is_available(),
	}
	train_dataloader = DataLoader(
		train_dataset,
		shuffle=True,
		generator=generator,
		**loader_options,
	)
	test_dataloader = DataLoader(test_dataset, shuffle=False, **loader_options)

	selected_targets = [full_train_dataset.targets[index] for index in selected_indices]
	class_counts = Counter(selected_targets)
	count_summary = ", ".join(
		f"{class_name}={class_counts[class_index]}"
		for class_index, class_name in enumerate(full_train_dataset.classes)
	)
	print(
		f"Training with {len(train_dataset)}/{len(full_train_dataset)} images "
		f"({train_percentage:g}% requested): {count_summary}"
	)
	print(f"Evaluating with all {len(test_dataset)} test images")

	return train_dataloader, test_dataloader, full_train_dataset.classes

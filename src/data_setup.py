import math
import random
from collections import Counter
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


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
) -> tuple[DataLoader, DataLoader, list[str]]:
	"""Create loaders using a balanced percentage of training data and all test data.

	Increasing ``train_percentage`` with the same seed preserves previously selected
	samples, which makes progressive experiments directly comparable.
	"""
	if not 0 < train_percentage <= 100:
		raise ValueError("train_percentage must be greater than 0 and at most 100")
	if batch_size < 1:
		raise ValueError("batch_size must be at least 1")

	full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
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

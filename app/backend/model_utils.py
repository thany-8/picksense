"""Model loading and image inference helpers for the PickSense API."""

from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


# ImageFolder sorted these folder names alphabetically during training.
CLASS_NAMES = ("clear", "heavily_occluded", "partially_occluded")


class InvalidImageError(ValueError):
	"""Raised when uploaded bytes cannot be decoded as a supported image."""


class PickSensePredictor:
	"""Own the trained model, preprocessing transform, and inference device."""

	def __init__(self, checkpoint_path: Path) -> None:
		if not checkpoint_path.is_file():
			raise FileNotFoundError(
				"PickSense checkpoint not found at "
				f"'{checkpoint_path}'. Download pretrained_vit_picksense.pth from "
				"Google Drive into models/, or set PICKSENSE_MODEL_PATH."
			)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		weights = ViT_B_16_Weights.DEFAULT
		self.transform = weights.transforms()

		# weights=None avoids downloading ImageNet weights. The checkpoint contains
		# the complete trained state_dict for both the backbone and classifier head.
		self.model = vit_b_16(weights=None)
		self.model.heads = nn.Linear(
			in_features=self.model.hidden_dim,
			out_features=len(CLASS_NAMES),
		)

		state_dict = torch.load(
			checkpoint_path,
			map_location=self.device,
			weights_only=True,
		)
		self.model.load_state_dict(state_dict)
		self.model.to(self.device)
		self.model.eval()

	def predict(self, image_bytes: bytes) -> dict[str, object]:
		"""Return the predicted class and all three softmax probabilities."""
		try:
			with Image.open(BytesIO(image_bytes)) as uploaded_image:
				image = uploaded_image.convert("RGB")
		except (UnidentifiedImageError, OSError) as error:
			raise InvalidImageError("The uploaded file is not a readable image.") from error

		image_tensor = self.transform(image).unsqueeze(0).to(self.device)

		with torch.inference_mode():
			logits = self.model(image_tensor)
			probability_tensor = torch.softmax(logits, dim=1)[0].cpu()

		probabilities = OrderedDict(
			(class_name, probability_tensor[index].item())
			for index, class_name in enumerate(CLASS_NAMES)
		)
		predicted_index = int(probability_tensor.argmax().item())
		prediction = CLASS_NAMES[predicted_index]

		return {
			"prediction": prediction,
			"probability": probabilities[prediction],
			"probabilities": probabilities,
		}
"""FastAPI application for PickSense image predictions."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .model_utils import InvalidImageError, PickSensePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "pretrained_vit_picksense.pth"
MODEL_PATH = Path(os.getenv("PICKSENSE_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
	"""Load the model once before requests are accepted."""
	app.state.predictor = PickSensePredictor(MODEL_PATH)
	yield


app = FastAPI(
	title="PickSense API",
	description="Classify object images by visual occlusion level.",
	version="1.0.0",
	lifespan=lifespan,
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
	allow_credentials=True,
	allow_methods=["GET", "POST"],
	allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
	predictor: PickSensePredictor = request.app.state.predictor
	return {"status": "ok", "device": str(predictor.device)}


@app.post("/predict")
async def predict(
	request: Request,
	file: Annotated[UploadFile, File(description="Object image to classify")],
) -> dict[str, object]:
	if file.content_type not in ALLOWED_CONTENT_TYPES:
		raise HTTPException(
			status_code=415,
			detail="Upload a JPEG, PNG, WebP, or BMP image.",
		)

	image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
	if not image_bytes:
		raise HTTPException(status_code=400, detail="The uploaded image is empty.")
	if len(image_bytes) > MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail="Images must be 10 MB or smaller.")

	predictor: PickSensePredictor = request.app.state.predictor
	try:
		return predictor.predict(image_bytes)
	except InvalidImageError as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
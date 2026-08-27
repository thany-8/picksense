# PickSense Web App

This app serves the trained PickSense ViT through FastAPI and provides a React
interface for image upload, preview, and occlusion probabilities.

## Project structure

```text
app/
├── backend/
│   ├── main.py             # FastAPI routes and startup lifecycle
│   ├── model_utils.py      # exact ViT reconstruction and inference
│   └── requirements.txt
└── frontend/
    ├── src/App.jsx         # upload and prediction interface
    ├── src/styles.css
    └── package.json
```

## 1. Download the trained checkpoint

The notebook saved the model state dictionary to Google Drive:

```text
/content/drive/MyDrive/PickSense/models/pretrained_vit_picksense.pth
```

Download that file and place it here:

```text
models/pretrained_vit_picksense.pth
```

The checkpoint is approximately 327 MB and is intentionally excluded from Git.
To keep it somewhere else, set `PICKSENSE_MODEL_PATH` to its path before starting
the API.

## 2. Start the backend

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r app/backend/requirements.txt
uvicorn app.backend.main:app --reload --port 8001
```

Check the API at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).
The model is reconstructed and loaded once when FastAPI starts.

## 3. Start the frontend

Open another terminal from the repository root:

```bash
cd app/frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

For a backend at another URL, copy `.env.example` to `.env` and update
`VITE_API_URL`.

## Single-origin option (one server)

For a quick end-to-end check without running a second dev server, build the
frontend once and let the backend serve it from the same origin:

```bash
cd app/frontend
npm run build
```

Then start the backend and open [http://127.0.0.1:8001](http://127.0.0.1:8001).
When `app/frontend/dist/` exists, the API serves the built UI at `/` (and its
assets under `/assets`), so uploads and predictions work with no CORS setup.
Rebuild after changing frontend source, or use the Vite dev server above when
you want hot reloading.

## Inference contract

The backend deliberately matches the notebook:

- Architecture: `torchvision.models.vit_b_16`
- Preprocessing: `ViT_B_16_Weights.DEFAULT.transforms()`
- Head: one `Linear(768, 3)` layer
- Class index order: `clear`, `heavily_occluded`, `partially_occluded`
- Inference: `model.eval()` and `torch.inference_mode()`

The class index order comes from `torchvision.datasets.ImageFolder`, which sorts
folder names alphabetically. The frontend presents the classes in the more
natural order clear, partial, heavy without changing their probabilities.
# PickSense

PickSense is a computer vision learning project that explores whether a Vision Transformer (ViT) can classify an object's **pickability from visual occlusion**.

## Visual overview (start here)

New to the project? Open [`docs/picksense_system_overview.excalidraw`](docs/picksense_system_overview.excalidraw)
for a one-page diagram of how everything fits together. It shows the two halves
of the system side by side:

1. **Training** — download data, build a balanced dataset, and train the ViT
   once to produce the `pretrained_vit_picksense.pth` model file.
2. **Inference** — the web app loads that model file and predicts the occlusion
   level of an uploaded image (it never retrains).

To view or edit it, drag the file onto [excalidraw.com](https://excalidraw.com)
(or use the *Excalidraw* VS Code extension).

## Project objective

The current objective is to classify an image into one of three occlusion-based pickability categories:

| Class | OpenLORIS tasks | Approximate occlusion |
|---|---|---:|
| `clear` | `task1`, `task2`, `task3` | 0% |
| `partially_occluded` | `task4`, `task5`, `task6` | 25% |
| `heavily_occluded` | `task7`, `task8`, `task9` | 50% |

Occlusion is used as a **proxy for pickability**:

- A clear object is assumed to be easier to identify and pick.
- A partially occluded object may be more difficult to pick.
- A heavily occluded object is assumed to be the most difficult to pick.

> This project currently predicts visual occlusion categories. It does not yet measure grasp geometry, physical reachability, collision risk, or real robotic grasp success.

## Dataset

The dataset is prepared by `notebooks/00_download_prepare_data.ipynb`, which downloads the OpenLORIS-Object dataset from Kaggle:

[OpenLORIS-Object Dataset](https://www.kaggle.com/datasets/zhedamai/openlorisobject)

The relevant images are taken from the dataset's `occlusion` condition. The original OpenLORIS train and test splits are preserved.

Everything is stored **permanently on Google Drive** (not under `/content/`, which is wiped when the Colab runtime disconnects):

```text
/content/drive/MyDrive/PickSense/data/
├── raw/openloris/occlusion/     # OpenLORIS occlusion subset (downloaded once)
└── picksense_mini/              # balanced 3,600-image dataset
    ├── train/{clear, partially_occluded, heavily_occluded}/   # 1,000 each
    └── test/{clear, partially_occluded, heavily_occluded}/    # 200 each
```

`picksense_mini` is a balanced dataset (3,000 train + 600 test = 3,600 images). Its images are **copied** (not symlinked) so they persist on Drive, and sampling uses `random.seed(42)` for reproducibility. Train images are drawn from `occlusion/train` and test images from `occlusion/test`, so no image is shared between the two splits.

## Current workflow

Dataset preparation and modelling are split into two notebooks so the dataset is downloaded **once** instead of on every runtime restart.

**`00_download_prepare_data.ipynb`** — run once, or whenever the data changes:

1. Mounts Google Drive.
2. Downloads OpenLORIS-Object with `kagglehub` **only if it is missing**.
3. Persists the `occlusion` subset to Drive so it survives runtime restarts.
4. Builds the balanced `picksense_mini` dataset **only if it is missing**.
5. Prints a verification summary (paths, per-class counts, totals, disk size).

**`01_picksense_main.ipynb`** — normal work; it **never downloads** anything:

1. Mounts Google Drive and checks that `picksense_mini` exists (otherwise it tells you to run `00_download_prepare_data.ipynb` first).
2. Installs and imports PyTorch dependencies.
3. Creates PyTorch `ImageFolder` datasets.
4. Resizes images to `224 × 224` and builds training/testing `DataLoader` objects.
5. Replicates the Vision Transformer (ViT) architecture from the Learn PyTorch course.
6. Trains and evaluates the model.

## Web application

The `app/` directory contains a React + Vite frontend and a FastAPI + PyTorch
inference backend. It loads the trained pretrained-ViT checkpoint once at API
startup and returns all three softmax probabilities for an uploaded image.

See [app/README.md](app/README.md) for checkpoint placement and local startup
commands. The web app performs inference only; it never retrains the model.

## Model approach

The project follows the ViT paper-replication approach from Learn PyTorch.

A ViT processes an image by:

1. Resizing the image to `224 × 224`.
2. Dividing it into fixed-size patches.
3. Converting each patch into an embedding.
4. Adding class and positional embeddings.
5. Processing the sequence with Transformer encoder blocks.
6. Using the class token to predict one of the three output classes.

The initial preprocessing pipeline is:

```python
manual_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
```

## Progressive data training

The main notebook can train on a deterministic, class-balanced percentage of
the prepared training set. Change these values in the DataLoader cell:

```python
TRAIN_PERCENTAGE = 100 # Use the complete prepared training set
BATCH_SIZE = 8         # Reduce to 4, 2, or 1 after a CUDA out-of-memory error
DATA_SEED = 42
```

Keep `DATA_SEED` fixed when comparing percentages. Each larger percentage then
contains every image selected by the smaller percentages. Evaluation always
uses 100% of the test set, so loss and accuracy remain comparable across runs.

Dataset percentage and GPU memory solve different problems: a larger percentage
adds more batches per epoch, while `BATCH_SIZE` determines the GPU memory needed
for one training step.

`picksense_mini` currently has 1,000 training images per class. A 100% run uses
all 3,000 training images. Lower percentages remain useful for quick experiments,
but final comparisons should use 100% of the prepared pool.

## Repository structure

```text
picksense/
├── notebooks/
│   ├── 00_download_prepare_data.ipynb   # run once: download + build picksense_mini on Drive
│   └── 01_picksense_main.ipynb          # normal work: checks data, then trains the ViT
├── src/
│   ├── create_mini_dataset.py           # standalone mini-dataset builder (CLI)
│   └── data_setup.py                    # balanced percentage DataLoaders
└── README.md
```

The dataset itself lives on Google Drive under `/content/drive/MyDrive/PickSense/data/` and is **not** committed to Git.

## Getting started

### Option 1: Google Colab

Google Colab is the simplest environment for running the notebooks.

Clone the repository in Colab:

```bash
git clone https://github.com/thany-8/picksense.git
cd picksense
```

Then, the first time (or whenever the data changes), run:

```text
notebooks/00_download_prepare_data.ipynb
```

For all normal work (training, evaluation), run:

```text
notebooks/01_picksense_main.ipynb
```

### Option 2: Local development

Clone the repository:

```bash
git clone https://github.com/thany-8/picksense.git
cd picksense
```

Create and activate a virtual environment on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the main dependencies:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio matplotlib pillow torchinfo kagglehub jupyter
```

Start Jupyter:

```bash
jupyter notebook notebooks/01_picksense_main.ipynb
```

The notebooks are written for Colab and use Google Drive paths (`/content/drive/MyDrive/PickSense/...`). To run locally, either build the dataset with `src/create_mini_dataset.py` and update the paths in the notebooks, or use the Colab option above.

## Planned work

- Complete the ViT architecture implementation.
- Train the model on the three occlusion classes.
- Track training and testing loss and accuracy.
- Add a confusion matrix and per-class metrics.
- Evaluate predictions on unseen images.
- Compare a custom ViT with a pretrained ViT model.
- Add data augmentation and normalization.
- Save trained model weights.
- Move reusable training and data-processing code into Python modules.
- Investigate labels based on real robotic grasp success rather than occlusion alone.

## Project status

This project is currently under development and is intended for learning and experimentation.

The dataset preparation and `DataLoader` pipeline are implemented. The ViT architecture, training process, and final evaluation are still being developed.

The OpenLORIS-Object dataset remains subject to its original license and terms of use.
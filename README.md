# PickSense

PickSense is a computer vision learning project that explores whether a Vision Transformer (ViT) can classify an object's **pickability from visual occlusion**.



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

The notebook downloads the OpenLORIS-Object dataset from Kaggle:

[OpenLORIS-Object Dataset](https://www.kaggle.com/datasets/zhedamai/openlorisobject)

The relevant images are taken from the dataset's `occlusion` condition. The original OpenLORIS train and test splits are preserved.

The notebook reorganizes the selected images into an `ImageFolder`-compatible structure:

```text
data/
└── pickability_occlusion/
    ├── train/
    │   ├── clear/
    │   ├── partially_occluded/
    │   └── heavily_occluded/
    └── test/
        ├── clear/
        ├── partially_occluded/
        └── heavily_occluded/
```

Symbolic links are used by default to avoid duplicating the image files.

## Current workflow

The notebook currently covers:

1. Installing and importing PyTorch dependencies.
2. Downloading OpenLORIS-Object with `kagglehub`.
3. Inspecting the dataset and sample images.
4. Locating the OpenLORIS occlusion subset.
5. Converting its tasks into three pickability labels.
6. Creating PyTorch `ImageFolder` datasets.
7. Resizing images to `224 × 224`.
8. Creating training and testing `DataLoader` objects.
9. Visualizing image batches and labels.
10. Studying and replicating the Vision Transformer architecture.

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

The current batch size is `32`.

## Repository structure

```text
picksense/
├── notebooks/
│   └── picksense.ipynb
├── data/
│   ├── raw/
│   │   └── openloris/
│   └── pickability_occlusion/
└── README.md
```

The `data` directories are created when the notebook is executed and should generally not be committed to Git.

## Getting started

### Option 1: Google Colab

The notebook currently uses paths beginning with:

```text
/content/picksense/
```

For that reason, Google Colab is the simplest environment for running the notebook in its current form.

Clone the repository in Colab:

```bash
git clone https://github.com/thany-8/picksense.git
cd picksense
```

Open and run:

```text
notebooks/picksense.ipynb
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
jupyter notebook notebooks/picksense.ipynb
```

When running locally, update the notebook's `/content/picksense/...` paths or replace them with paths relative to the repository root.

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
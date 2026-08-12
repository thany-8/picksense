# PickSense
**A computer vision project that uses CNNs and Vision Transformers to study visual accessibility for robotic perception — how well a model can recognize the visual conditions (starting with occlusion) that make an object harder for a robotic vision system to perceive.**

---

## The Problem

Before a robot can identify, track, or interact with an object, it first has to **see that object clearly enough**.

In real environments this becomes difficult when an object is:

* partially hidden by another object,
* surrounded by visual clutter,
* very small in the camera view,
* or captured under poor lighting.

PickSense focuses on this **perception step before robotic manipulation**. Rather than predicting grasp coordinates or controlling a robotic arm, it studies whether a vision model can recognize the visual conditions that make an object easier or harder for a robot to perceive.

The central question behind the project:

> **How well can CNNs and Vision Transformers recognize changes in object visibility, especially as occlusion increases?**

---

## What This Project Builds

PickSense is an **image-classification project**. The first version focuses on **occlusion-level classification**.

| Input | Prediction |
| ------------------------------------ | ------------------ |
| RGB image containing a target object | No / low occlusion |
| | Partial occlusion  |
| | Heavy occlusion    |

The OpenLORIS-Object dataset grades occlusion into **9 ordered levels** (`task1`–`task9`). PickSense starts by grouping these into a few coarse bands so the first experiment stays a clean, well-posed classification problem; the graded levels can also be used directly in later experiments.

Future experiments may extend the system to additional visual-accessibility factors already labeled in the dataset:

* surrounding clutter,
* apparent object size,
* illumination.

These factors describe the quality of the robot's *visual access* to an object rather than whether the object is physically graspable.

---

## Dataset

PickSense uses the **OpenLORIS-Object** robotic-vision dataset, downloaded via `kagglehub` (`zhedamai/openlorisobject`, ~18 GB).

It contains images of everyday household objects (cups, bowls, bottles, scissors, toys, …) captured under controlled, pre-labeled environmental conditions, organized into factor folders:

| Factor | Folder | What it varies |
| ------------ | --------------- | ------------------------------------------------------------ |
| Occlusion | `occlusion/` | how much of the object is hidden — 9 graded levels (`task1`–`task9`) |
| Clutter | `clutter/` | amount of surrounding visual clutter |
| Object size | `pixel/` | how large the object appears in the frame |
| Illumination | `illumination/` | lighting strength |
| Combined | `sequence/` | several factors varied together |

Each factor is split into `train` / `test` (with `sequence` also providing `validation`).

A major reason for choosing this dataset is that these environmental conditions **already have defined levels**, allowing PickSense to use existing labels instead of creating subjective pickability labels from raw robotics metadata. For the first experiment, PickSense focuses only on **occlusion** so that the project remains a clear image-classification problem. Additional factors can be introduced after the initial pipeline works.

---

## Why This Design

### Keep the problem focused

PickSense does not attempt to solve grasp planning, object pose estimation, or robot control:

```text
Camera image
    ↓
Visual perception
    ↓
PickSense  →  assess viewing condition (e.g. occlusion level)
    ↓
Downstream robotic system
```

This keeps the project focused on computer vision while still connecting it to a meaningful robotics problem.

### Use existing labels

The original version of PickSense required creating custom labels from visibility measurements, object masks, and scene statistics. The redesigned project uses a dataset with **defined environmental-condition labels**, making the training target easier to understand and evaluate.

### Learn Vision Transformers deeply

One of the main goals of the project is to implement a Vision Transformer **from scratch in PyTorch** while following the architecture introduced in the ViT paper. The project therefore focuses on understanding:

* image patches,
* patch embeddings,
* positional embeddings,
* the class token,
* multi-head self-attention,
* transformer encoder blocks,
* and the classification head.

---

## Approach

The work is developed through Jupyter notebooks that follow the complete machine-learning workflow, supported by small helper modules in `src/`.

### 1. Data Exploration

Explore the images and labels and understand how different occlusion levels affect object visibility.

### 2. Data Preparation

Build train, validation, and test sets and apply the image transformations each model expects.

### 3. CNN Baseline

Train a small convolutional neural network to establish a baseline. The CNN provides a reference point for determining whether the Vision Transformer offers an improvement.

### 4. Vision Transformer From Scratch

Implement a compact Vision Transformer in PyTorch, including:

```text
Image
  ↓
Patches
  ↓
Patch Embeddings
  ↓
Class Token + Positional Embeddings
  ↓
Transformer Encoder
  ↓
Classification Head
  ↓
Occlusion level
```

### 5. Model Comparison

Compare the CNN and Vision Transformer using metrics such as:

* accuracy,
* precision,
* recall,
* macro F1-score,
* confusion matrix.

### 6. Transfer Learning

Fine-tune a pretrained Vision Transformer and compare it with the model trained from scratch.

### 7. Error Analysis

Inspect incorrect predictions to understand which visual situations are most difficult for each architecture. Potential extensions include attention visualization, confidence calibration, clutter classification, and object-scale classification.

---

## Key Research Questions

* **Can a CNN or Vision Transformer recognize the visual conditions that make an object harder for a robotic vision system to perceive?** (the central question)
* How accurately can a vision model recognize different levels of object occlusion?
* How does a CNN compare with a Vision Transformer on this task?
* Does a ViT trained from scratch perform competitively on this dataset?
* How much does pretrained Vision Transformer knowledge improve performance?
* Which types of occlusion cause the most classification errors?
* Does model confidence decrease as visual access to the object becomes more difficult?

Future experiments may also investigate:

* How does surrounding clutter affect model performance?
* How does apparent object size affect recognition?
* Are Vision Transformers more robust than CNNs as viewing conditions become more difficult?

---

## Example Prediction

```text
Input:
RGB image of an object

PickSense Assessment
Occlusion level:  Partial
Confidence:       89%
```

A future, multi-factor version could provide a broader visual assessment:

```text
Visual Accessibility
Occlusion:     Partial
Clutter:       Complex
Object scale:  Medium
Confidence:    87%
```

These predictions describe **visual conditions only** and should not be interpreted as guarantees of robotic grasp success.

---

## Models

The project will compare:

| Model | Purpose |
| ------------------------------- | ----------------------------------------- |
| Small CNN | Baseline |
| Vision Transformer from scratch | Understand and reproduce the ViT architecture |
| Pretrained Vision Transformer | Study transfer learning |

The main comparison is:

```text
CNN
  vs.
ViT from scratch
  vs.
Pretrained ViT
```

---

## Tech Stack

**Machine learning**

* Python
* PyTorch
* TorchVision
* NumPy
* pandas
* scikit-learn
* Pillow, SciPy

**Data & experimentation**

* kagglehub (dataset download)
* Jupyter / Google Colab
* Matplotlib

**Dataset**

* OpenLORIS-Object

---

## Repository Structure

```text
picksense/
├── data/
│   ├── raw/          # OpenLORIS-Object (downloaded via kagglehub)
│   ├── processed/    # occlusion train / val / test splits
│   └── samples/      # small image samples for quick checks
├── notebooks/
│   └── picksense.ipynb    # Implementation from scratch
├── src/              # helper modules: data_setup, engine, model, train, utils
├── models/           # saved checkpoints
├── reports/          # figures, metrics, confusion matrices
├── requirements.txt
├── README.md
└── .gitignore
```

The workflow is notebook-driven — a data-download notebook plus a main experiment notebook — with reusable logic factored into `src/`, documenting the complete learning and experimentation process from dataset exploration through model evaluation.

---

## Project Roadmap

* [ ] Explore the OpenLORIS-Object dataset
* [ ] Prepare the occlusion-classification dataset
* [ ] Visualize and verify class labels
* [ ] Build train / validation / test sets
* [ ] Train the CNN baseline
* [ ] Implement the Vision Transformer from scratch
* [ ] Compare CNN vs. custom ViT
* [ ] Fine-tune a pretrained ViT
* [ ] Evaluate using accuracy, macro F1, and a confusion matrix
* [ ] Perform error analysis
* [ ] Add attention visualization
* [ ] Experiment with clutter or object-scale classification

---

## Current Status

🚧 **Dataset exploration and pipeline design.**

Current milestone:

> Prepare the labeled occlusion dataset and establish the CNN baseline before beginning the Vision Transformer implementation.

---

## Limitations

PickSense is a computer-vision research and portfolio project, not a complete robotic-manipulation system. It does **not**:

* control a robotic arm,
* calculate grasp coordinates,
* predict physical grasp success,
* account for object weight, material, or stability,
* guarantee that a visually accessible object can be safely picked.

Instead, PickSense focuses specifically on the **visual perception conditions that occur before robotic interaction**.

---

## Project Goal

PickSense combines a practical robotic-vision problem with a from-scratch implementation of the Vision Transformer architecture. The goal is not only to achieve good classification performance, but also to understand **when and why CNNs and Vision Transformers behave differently as visual access to an object becomes more challenging**.

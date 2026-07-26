# PickSense
**A Vision Transformer-based system that predicts whether an object in a cluttered warehouse bin is safe and practical for a robot to pick.**

---

## The Problem

Before a warehouse robot reaches into a bin, it has to answer a question that's easy to state and hard to solve visually:

> *"Can I confidently pick this object, or should I reposition the camera, choose a different object, or ask a human for help?"*

Most grasping research jumps straight to *how* to pick (grasp pose estimation, robotic arm control). PickSense focuses on the step before that: the **visual pickability decision**. Given an image of a cluttered bin, the system judges whether it's safe and sensible to attempt a pick right now — before any physical action is taken.

This matters in practice. A failed pick attempt in a real fulfillment center isn't just a missed grasp — it can damage products, drop items, slow down operations, or force a repeated action. A visual check before committing to a grasp is a cheap, high-value safety layer.

## What This Project Builds

Not a full robotic pipeline (no arm control, no physical grasping). Just the **perception + decision component**, built as an image-classification problem:

| Input | Output |
|---|---|
| Image of a warehouse bin | Pickability category: **Easy Pick / Difficult Pick / Unsafe Pick / Human Assistance Recommended** |
| | Confidence score |
| | Likely reason for difficulty: occlusion, clutter, overlap, low visibility |
| | Visual explanation (attention/saliency map) |

## Why This Design

- **Decoupled from grasp planning** — this component sits in front of any downstream grasp-pose estimator; it doesn't compute exact grasp coordinates or control hardware.
- **Risk-aware, not just accuracy-aware** — a false "Easy Pick" (dangerous) is treated as more costly than a false "Difficult Pick" (just overly cautious). The evaluation is built around that asymmetry, not raw accuracy.
- **Human-in-the-loop by design** — "Human Assistance Recommended" is a first-class output, not a failure case, reflecting how automation actually gets deployed on warehouse floors.
- **Explainable by design** — attention/saliency maps let a human quickly sanity-check *why* the model made a call, which matters for trust in a safety-adjacent system.

## Approach

1. **Dataset**: A synthetic robotic bin-picking dataset (RGB images, object masks, poses, visibility measurements, multiple clutter levels). Visibility and scene information are used to derive pickability labels — treated as project-defined labels rather than absolute ground truth, refined against a manually reviewed sample. A small real-world photo set (household packages in a bin) may be added later to study synthetic-to-real generalization.

2. **Models compared**:
   - CNN baseline (trained from scratch)
   - Pretrained CNN (ResNet/EfficientNet, fine-tuned)
   - Custom Vision Transformer, implemented from scratch in PyTorch (patch embeddings, positional embeddings, multi-head self-attention, transformer encoder blocks)
   - Pretrained ViT (fine-tuned)
   - *(Stretch goal)* Multi-task ViT — one shared encoder with a pickability head and a difficulty-reason head

3. **Training pipeline**: dataset exploration → label generation → train/val/test splits → augmentation → class-imbalance handling → training with AdamW, LR scheduling, early stopping → checkpointing → error analysis → export.

4. **Evaluation**: accuracy alone isn't sufficient. The project tracks precision/recall, macro F1, confusion matrix, AUROC, calibration error, inference latency, and — most importantly — the **cost-weighted split between false-safe and false-warning predictions**.

## Key Research Questions

- Does a Vision Transformer outperform a CNN on cluttered bin scenes, or does clutter defeat both similarly?
- Does transfer learning beat training a ViT from scratch on a domain this narrow?
- Which visual conditions (heavy occlusion, stacked similar objects, low light) drive the most errors?
- How well-calibrated are the model's confidence scores — does 90% confidence actually mean 90% correct?
- How much real-world data is needed to close the synthetic-to-real gap?

## Deployment Plan

A FastAPI inference service behind a lightweight web app (Streamlit or React). A user uploads a bin image and gets back: the predicted category, class probabilities, an attention/saliency overlay, and a recommended action (continue / adjust approach / select another object / request human help).

## Tech Stack

**ML**: PyTorch, TorchVision, Hugging Face, NumPy, pandas, scikit-learn
**Experimentation**: Weights & Biases / MLflow, Jupyter, TensorBoard
**Backend**: FastAPI, Pydantic
**App**: Streamlit or React
**Deployment**: Docker, GitHub Actions

## Repo Structure
```
picksense/
├── data/              # raw, processed, real-world test images
├── notebooks/         # exploration, labeling analysis, error analysis
├── src/
│   ├── data/          # dataset, preprocessing, labeling
│   ├── models/         # cnn, custom_vit, pretrained_vit, multitask_vit
│   ├── training/       # train, evaluate, losses
│   ├── inference/      # predict, explain (attention maps)
│   └── utils/
├── app/               # api + frontend
├── reports/           # figures, experiment results
├── Dockerfile
└── README.md
```

## Status
🚧 **Planning & dataset evaluation.**
Next milestone: build the data pipeline and train the CNN baseline.

## Roadmap
- [ ] Dataset exploration + pickability labeling rules
- [ ] CNN baseline
- [ ] Custom ViT built from scratch + comparison vs. CNN
- [ ] Pretrained ViT fine-tuning
- [ ] Confidence calibration + attention visualization
- [ ] Synthetic-to-real generalization test
- [ ] FastAPI + demo app deployment

## Limitations

PickSense is a research/portfolio prototype, not a production safety system. It does not control a physical robot, does not compute exact grasp coordinates, and does not account for object weight or material. Synthetic training data may not fully represent real warehouses, and attention maps offer partial — not complete — explanations of model reasoning. Human review remains important for uncertain or high-risk cases.

---

*This project explores the perception layer of warehouse robotics — the visual judgment call a robot makes before it acts.*

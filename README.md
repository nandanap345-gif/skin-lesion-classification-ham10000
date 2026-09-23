# Skin Lesion Classification — HAM10000

A comparative deep learning project for multiclass skin lesion classification, built and trained on the **HAM10000** dermoscopic image dataset (10,015 images, 7 diagnostic classes). Three hybrid CNN–attention and CNN–Transformer architectures are implemented from scratch in PyTorch, trained under an identical protocol, and evaluated with a full explainability and uncertainty-estimation pipeline.

## Overview

Skin cancer diagnosis from dermoscopic images is a fine-grained, clinically imbalanced classification problem. This project explores whether hybridizing convolutional backbones with attention mechanisms (CBAM) or Transformer encoders improves performance over either component alone, while also comparing model size and training efficiency — a practical concern for real-world deployment.

**Classes:** `akiec` (actinic keratoses) · `bcc` (basal cell carcinoma) · `bkl` (benign keratosis-like lesions) · `df` (dermatofibroma) · `mel` (melanoma) · `nv` (melanocytic nevi) · `vasc` (vascular lesions)

## Architectures

| Model | Backbone | Attention / Sequence Modeling | Parameters |
|---|---|---|---|
| ResNet50 + Transformer | ResNet-50 (ImageNet pretrained) | 2-layer, 8-head Transformer encoder | 25.09M |
| EfficientNet-B3 + CBAM | EfficientNet-B3 (ImageNet pretrained) | CBAM (channel + spatial attention) | 11.00M |
| EfficientNet-B0 + Transformer | EfficientNet-B0 (ImageNet pretrained) | Sinusoidal positional encoding + 2-layer, 8-head Transformer encoder | 5.39M |

All three models feed CNN feature maps into their respective attention/sequence module, followed by global pooling and a linear classification head.

## Key Techniques

- **CBAM** (Convolutional Block Attention Module) — channel and spatial attention, implemented from scratch
- **Multi-head self-attention / Transformer encoders** over flattened CNN feature maps
- **Custom sinusoidal positional encoding** (sin/cos, as in the original Transformer paper)
- **Grad-CAM** — implemented from scratch using PyTorch forward/backward hooks for visual explainability
- **Monte Carlo Dropout** — 20 stochastic forward passes at inference to estimate prediction confidence and uncertainty
- **Class-weighted Cross-Entropy loss** with label smoothing (0.05) to handle severe class imbalance (`nv` = 67% of the dataset)
- **Transfer learning** from ImageNet-pretrained backbones
- **AdamW** optimizer with **ReduceLROnPlateau** scheduling (monitored on validation Macro-F1)

## Results

Final test-set performance after 15 epochs of training per model:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Accuracy | Train Time (min) |
|---|---|---|---|---|---|---|
| ResNet50 + Transformer | 0.8536 | 0.7025 | 0.8201 | 0.7532 | 0.8201 | 26.92 |
| EfficientNet-B3 + CBAM | 0.8403 | 0.7437 | 0.7923 | 0.7561 | 0.7923 | 26.58 |
| **EfficientNet-B0 + Transformer** | **0.8576** | 0.7346 | 0.8078 | **0.7658** | 0.8078 | **23.03** |

**EfficientNet-B0 + Transformer** achieves the best accuracy and Macro F1 while being the smallest and fastest-training model — roughly a fifth the parameter count of ResNet50 + Transformer.

All models score highest on the majority class `nv` (F1 > 0.91) and lowest on `mel` (melanoma, F1 0.64–0.68) — the most clinically critical class to classify correctly, and the clearest target for future improvement (e.g. targeted oversampling, stronger augmentation).

## Pipeline

```
HAM10000 Dataset → Preprocessing & Augmentation → Stratified 70/15/15 Split
→ Train 3 CNN-Hybrid Models (class-weighted loss) → Evaluate on Test Set
→ Explainability (Grad-CAM, Monte Carlo Dropout)
```

- Images resized to 224×224, ImageNet-normalized
- Training augmentation: random horizontal/vertical flips, ±20° rotation, colour jitter
- Stratified split: 7,010 train / 1,502 validation / 1,503 test images

## Tech Stack

Python · PyTorch · torchvision · scikit-learn · NumPy · pandas · Matplotlib · Seaborn · PIL · Google Colab (GPU)

## Repository Structure

```
├── DL_PROJECT.ipynb        # Full notebook: data pipeline, models, training, evaluation
├── README.md
└── outputs/                 # (optional) saved figures, model checkpoints
```

## How to Run

1. Clone the repository and open `DL_PROJECT.ipynb` in Google Colab or Jupyter.
2. Download the [HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi.org/10.7910/DVN/DBW86T) and update the dataset path in the notebook.
3. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn pandas matplotlib seaborn pillow
   ```
4. Run all cells sequentially. Each model trains for 15 epochs (GPU recommended — CPU training is significantly slower).

## Future Work

- Confusion-matrix-based error analysis to identify specific class confusions
- k-fold cross-validation for more robust generalization estimates
- Ensembling all three models to combine complementary precision/recall strengths
- Targeted oversampling or synthetic augmentation for underperforming classes (`mel`, `akiec`)
- Validation on an independent external dermoscopic dataset

## License

MIT

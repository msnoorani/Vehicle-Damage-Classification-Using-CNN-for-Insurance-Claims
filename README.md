# Vehicle Damage Classification Using CNN for Insurance Claims

> **Understanding AI Module | University of Hull**  
> Author: Muhammad Salahuddin

---

## 📌 Project Overview

This project develops a **Convolutional Neural Network (CNN)** to automatically classify vehicle damage from images across 5 damage categories, designed to assist insurance companies in streamlining and automating their claims verification process.

The model acts as an **initial screening layer** — classifying damage type before human adjuster review, reducing bottlenecks and operational costs in traditional manual inspection workflows.

---

## 🏷️ Damage Categories

| Class | Training Samples |
|-------|-----------------|
| Scratch | 384 |
| Dent | 340 |
| Window Broken | 189 |
| Lamp Broken | 169 |
| Flat Tire | 70 |

> Note: Significant class imbalance existed (Flat Tire: 70 vs Scratch: 384), addressed through data augmentation.

---

## 📊 Results

### Overall Performance
- **Training Accuracy:** ~65%
- **Validation Accuracy:** 62.67%
- **Train/Val gap:** 2–3% — confirming minimal overfitting

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Scratch | 0.60 | 0.67 | 0.63 |
| Flat Tire | **0.87** | 0.74 | 0.80 |
| Dent | 0.56 | 0.45 | 0.50 |
| Window Broken | 0.69 | **0.95** | 0.80 |
| Lamp Broken | 0.59 | 0.47 | 0.53 |

**Key Finding:** Window breakage detection achieved 95% recall — correctly identifying 180 out of 189 broken window cases. Dent classification was the most challenging due to visual similarity with scratches.

---

## 🏗️ CNN Architecture

| Layer | Parameters | Output Shape | Activation |
|-------|-----------|-------------|------------|
| Input | — | 224×224×3 | — |
| Conv2D | 32 filters (3×3) | 224×224×32 | ReLU |
| MaxPooling2D | 2×2 pool | 112×112×32 | — |
| Conv2D | 128 filters (3×3) | 112×112×128 | ReLU |
| MaxPooling2D | 2×2 pool | 56×56×128 | — |
| Conv2D | 128 filters (3×3) | 56×56×128 | ReLU |
| MaxPooling2D | 2×2 pool | 28×28×128 | — |
| Flatten | — | 100,352 | — |
| Dense | 256 units | 256 | ReLU |
| Dropout | 0.5 rate | 256 | — |
| Output | 5 units | 5 | Softmax |

**Design decisions:**
- Progressive filter increase (32→128→128) to capture hierarchical damage features
- MaxPooling to reduce spatial dimensions while preserving critical features
- ReLU activation to prevent vanishing gradients
- Dropout (p=0.5) to mitigate overfitting on limited dataset

---

## 🔧 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimiser | SGD (momentum=0.9) |
| Learning Rate | 0.01 |
| Batch Size | 32 |
| Epochs | 20 |
| Loss Function | Categorical Cross-Entropy |
| Train/Val Split | 80/20 |

**Data Augmentation applied:**
- Random rotations up to 20°
- Width/height shifts up to 10%
- Horizontal flips
- Zoom range 10%
- Shear range 10%

---

## 🧪 Preprocessing Pipeline

```
Raw Images (variable sizes)
  → Resize to 224×224 pixels
  → Normalise pixel values (÷255 → range [0,1])
  → 80/20 train-validation split
  → Data augmentation on training set
  → One-hot encode labels (5 classes)
```

---

## 💡 Key Findings

1. **Window damage** is the most reliably detected class (recall=0.95) — distinctive shattering patterns are easily learned by CNN
2. **Dents and scratches** are frequently confused due to visual similarity under varying lighting/angles (dent F1=0.50)
3. **Flat tires** showed highest precision (0.87) despite fewest training samples — distinctive shape features compensate for data scarcity
4. **Regularisation worked** — the 2-3% train/val accuracy gap confirms dropout + augmentation successfully prevented overfitting

---

## 🚀 Future Improvements

- Transfer learning with **ResNet or EfficientNet** for higher accuracy
- Expanded dataset for underrepresented classes (especially flat tires)
- Dedicated sub-models for visually similar categories (dents vs scratches)
- Integration with claim metadata (vehicle make/model/year) for context-aware classification

---

## 🛠️ Tech Stack

- **Python** — NumPy, Pandas, Matplotlib, PIL
- **Deep Learning** — TensorFlow/Keras (Sequential API)
- **Layers** — Conv2D, MaxPooling2D, Dense, Dropout, Flatten
- **Training** — SGD optimiser, ImageDataGenerator, EarlyStopping, ModelCheckpoint
- **Evaluation** — Classification report, Confusion matrix, Training curves

---

## 📁 Repository Structure

```
├── CNN_vehicle_damage.ipynb     # Full implementation notebook
├── README.md                    # Project documentation
└── data/
    └── README.md                # Dataset download instructions
```

---

## 📥 Dataset

This project uses the **Vehicle Damage Insurance Verification** dataset from Kaggle.

🔗 [Download from Kaggle](https://www.kaggle.com/datasets/sudhanshu2198/ripik-hackfest)

After downloading, place files as:
```
data/
├── train/
│   ├── images/
│   └── train.csv
└── test/
    ├── images/
    └── test.csv
```

---

## 👤 Author

**Muhammad Salahuddin**  
MSc Artificial Intelligence & Data Science — University of Hull  
[GitHub](https://github.com/msnoorani) | [LinkedIn](https://linkedin.com/in/msnoorani)

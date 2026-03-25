# Tomato Maturity Index Prediction and Disease Classification

**Author:** Williams Adaji-Agbane  


## Overview
A multi-model deep learning framework for dual-task tomato image classification:
- **Task A1** — Maturity Detection (Immature vs Mature) — Binary
- **Task A2** — Quality Grading (Fresh vs Rotten) — Binary  
- **Task B** — Disease Classification (10 classes) — Multi-class

## Models
| Model | Type | A1 Acc | A2 Acc | B Acc |
|-------|------|--------|--------|-------|
| TomatoNet | Custom Dual-Stream CNN | 98.67% | 92.95% | 88.35% |
| CustomCNN2 | Custom Inception-Residual CNN | 97.33% | 92.62% | 84.50% |
| DenseNet-121 | Pretrained Transfer Learning | 98.67% | 95.30% | 91.97% |
| HybridNet | Feature-Level Ensemble | 98.67% | 85.23% | 92.75% |
| PCA-LR | PCA + Logistic Regression | 98.67% | 94.30% | 87.43% |

## Datasets
- **Task A:** Sher-e-Bangla Tomato Maturity Detection and Quality Grading Dataset
  (Kaggle: sujaykapadnis/tomato-maturity-detection-and-quality-grading)
- **Task B:** PlantVillage Tomato Leaf Disease Dataset
  (Kaggle: charuchaudhry/plantvillage-tomato-leaf-dataset)

## Repository Structure
```
notebooks/    — Kaggle training notebook
results/      — Training logs (CSV) and result figures
```

## Environment
- Platform: Kaggle Notebooks (2× NVIDIA Tesla T4)
- Framework: TensorFlow 2.16 / Keras 3
- Python: 3.10

## How to Run
1. Upload the notebook to Kaggle
2. Add both datasets as inputs (see Dataset links above)
3. Set accelerator to GPU T4 x2
4. Run all cells sequentially
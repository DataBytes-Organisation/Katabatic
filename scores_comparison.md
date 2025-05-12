# TableGAN: Score Comparison

This document provides a direct comparison between the scores reported in the paper "Data Synthesis based on Generative Adversarial Networks" by Park et al. and our implementation results.

## Classifier Performance Comparison

### TSTR (Train on Synthetic, Test on Real)

| Classifier | Metric | Paper Results | Our Implementation |
|------------|--------|---------------|-------------------|
| **Random Forest** | Real Accuracy | ~84% | 85.48% |
| | Synthetic Accuracy | ~79% | 65.34% |
| | Accuracy Ratio | ~94% | 76.43% |
| **Logistic Regression** | Real Accuracy | ~84% | 82.26% |
| | Synthetic Accuracy | ~77% | 75.13% |
| | Accuracy Ratio | ~92% | 91.34% |
| **MLP** | Real Accuracy | ~82% | 84.93% |
| | Synthetic Accuracy | ~75% | 66.83% |
| | Accuracy Ratio | ~91% | 78.69% |
| **XGBoost** | Real Accuracy | N/A | 87.34% |
| | Synthetic Accuracy | N/A | 65.23% |
| | Accuracy Ratio | N/A | 74.68% |

## Statistical Similarity

| Metric | Paper Results | Our Implementation |
|--------|---------------|-------------------|
| Correlation Matrix Distance | ~0.30-0.35 | 5.94 |

## Implementation Details

- **Dataset**: Adult dataset
  - Paper: ~48,000 records
  - Our implementation: 45,222 records (after cleaning)
  
- **Training Parameters**:
  - Epochs: 200
  - Batch size: 500
  - Hidden dimensions: 100
  - Learning rate: 0.0002
  - Z dimension: 100
  - Train/test split: 70/30 
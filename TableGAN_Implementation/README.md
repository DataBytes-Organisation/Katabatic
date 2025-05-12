# TableGAN Implementation

This directory contains an implementation of the TableGAN model based on the paper "Data Synthesis based on Generative Adversarial Networks" by Park et al.

## Overview

TableGAN is a GAN-based model designed to generate synthetic tabular data while preserving the statistical properties and machine learning utility of the original data.

## Implementation Details

This implementation includes:

- Paper-exact implementation of the TableGAN architecture
- Evaluation metrics as described in the paper
- Comparison with the original paper results
- Multiple classifier evaluations (Random Forest, Logistic Regression, MLP, XGBoost)

## Results Summary

The implementation was evaluated on the Adult dataset with the following results:

### Statistical Similarity
- Correlation matrix Euclidean distance: 5.9404

### Machine Learning Efficacy (TSTR - Train on Synthetic, Test on Real)
- Random Forest:
  - Real data accuracy: 0.8548
  - Synthetic data accuracy: 0.6534
  - Accuracy ratio: 0.7643

- Logistic Regression:
  - Real data accuracy: 0.8226
  - Synthetic data accuracy: 0.7513
  - Accuracy ratio: 0.9134

- MLP:
  - Real data accuracy: 0.8493
  - Synthetic data accuracy: 0.6683
  - Accuracy ratio: 0.7869

- XGBoost:
  - Real data accuracy: 0.8734
  - Synthetic data accuracy: 0.6523
  - Accuracy ratio: 0.7468

## Files Included

- `paper_exact_implementation.py`: Main implementation of the TableGAN model
- `paper_replication.py`: Code to replicate the paper's results
- `run_paper_replication.py`: Runner script for the paper replication
- `run_adult_evaluation.py`: Evaluation script for the Adult dataset
- `results_summary.md`: Detailed summary of results
- `scores_comparison.md`: Comparison with paper results
- `model_links.md`: Correct links to GitHub repositories and papers for all models
- `requirements.txt`: Dependencies required for running the code
- `results/`: Directory containing output files, images, and detailed evaluation results

## GitHub Repository

The implementation is available on GitHub at:
[https://github.com/khushic/tableGAN-implementation](https://github.com/khushic/tableGAN-implementation)

## Usage

To run the TableGAN implementation and evaluation:

```bash
python run_paper_replication.py
```

## Dependencies

Required dependencies are listed in `requirements.txt`. 
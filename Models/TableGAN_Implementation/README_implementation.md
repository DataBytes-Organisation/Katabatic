# TableGAN Implementation

This is an implementation of TableGAN based on the paper:
**"Data Synthesis based on Generative Adversarial Networks"** by Noseong Park, Mahmoud Mohammadi, Kendra Gorde, Sushil Jajodia, Hongkyu Park, and Youngmin Kim (VLDB 2018).

Paper URL: [http://www.vldb.org/pvldb/vol11/p1071-park.pdf](http://www.vldb.org/pvldb/vol11/p1071-park.pdf)

## About TableGAN

TableGAN is a synthetic data generation technique based on Generative Adversarial Network (GAN) architecture. The goal is to protect sensitive data against re-identification attacks by producing synthetic data that preserves statistical features of the original data.

Key components of TableGAN include:
- A generator that produces synthetic table data
- A discriminator that distinguishes between real and synthetic data
- A classifier that ensures the synthetic data maintains the same predictive power

## Paper Implementation Details

This implementation follows the exact setup described in the paper, including:

1. **Dataset**: Adult income dataset from the UCI repository
2. **Architecture**: 
   - Generator with 3 fully connected layers
   - Discriminator with 3 fully connected layers
   - Classifier with 2 fully connected layers
3. **Training parameters**:
   - 200 epochs
   - Batch size of 500
   - Adam optimizer with learning rate 0.0002
4. **Privacy parameters**:
   - TEST_ID = 'OI_11_00': beta=1.0, delta_v=0.0, delta_m=0.0 (highest utility)
   - TEST_ID = 'OI_11_11': beta=1.0, delta_v=0.1, delta_m=0.1 (medium privacy-utility balance)
   - TEST_ID = 'OI_11_22': beta=1.0, delta_v=0.2, delta_m=0.2 (highest privacy)
5. **Evaluation metrics**:
   - Statistical similarity (correlation matrix distance)
   - Machine learning efficacy (comparing classifier performance)
   - Privacy protection (through membership inference attacks)

## Requirements

```
tensorflow==1.15.0
numpy>=1.16.0
pandas>=0.24.0
matplotlib>=3.0.0
seaborn>=0.9.0
scikit-learn>=0.20.0
scipy>=1.2.0
```

## How to Run

1. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the implementation script:
   ```
   python implementation.py
   ```

The script will:
- Train the TableGAN model for 200 epochs
- Generate synthetic data
- Evaluate the results on multiple metrics
- Save visualization plots and statistical comparisons

## Interpreting Results

The implementation produces several outputs:

1. **Correlation matrices comparison**:
   - Visualizes correlation matrices of real vs. synthetic data
   - Lower Euclidean distance between matrices indicates better statistical similarity

2. **Machine learning efficacy**:
   - Compares accuracy of classifiers trained on real vs. synthetic data
   - Higher accuracy ratio (synthetic/real) indicates better utility preservation
   - According to the paper, TableGAN typically achieves 90-95% of the real data accuracy

3. **Privacy-utility tradeoff**:
   - The paper explores different configurations (OI_11_00, OI_11_11, OI_11_22)
   - Higher privacy (OI_11_22) typically results in lower utility
   - The default configuration (OI_11_00) prioritizes utility over privacy

## Benchmarking Against Paper Results

For the Adult dataset, the paper reports:

1. **Statistical similarity**:
   - Correlation matrix distance: ~0.30-0.35

2. **Machine learning efficacy**:
   - Real data accuracy: ~84%
   - Synthetic data accuracy: ~79%
   - Accuracy ratio: ~0.94

3. **Privacy protection**:
   - Membership inference attack accuracy decreases as delta_v and delta_m increase
   - OI_11_22 provides the best privacy protection

Our implementation should achieve similar results, validating the paper's findings.

## Further Explorations

To explore different configurations:
1. Change the TEST_ID in implementation.py (options: 'OI_11_00', 'OI_11_11', 'OI_11_22')
2. Re-run the implementation to compare privacy-utility tradeoffs 
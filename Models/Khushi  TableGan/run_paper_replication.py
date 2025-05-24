#!/usr/bin/env python
"""
This script runs the complete TableGAN replication process:
1. Download and prepare the Adult dataset as described in the paper
2. Train the TableGAN model with paper-specified hyperparameters
3. Generate synthetic data and evaluate its quality

To run: python run_paper_replication.py
"""
import os
import subprocess
import time

def main():
    print("===== Starting TableGAN Paper Replication =====")
    
    # Step 1: Data preparation
    print("\nStep 1: Preparing the Adult dataset...")
    try:
        import paper_replication
        paper_replication.download_and_prepare_adult()
        print("✓ Dataset preparation completed successfully")
    except ImportError:
        print("Error: paper_replication.py not found. Running as separate process...")
        try:
            subprocess.run(["python", "paper_replication.py"], check=True)
            print("✓ Dataset preparation completed successfully")
        except subprocess.CalledProcessError:
            print("✗ Error: Failed to prepare dataset")
            return
    
    # Step 2: Train TableGAN model
    print("\nStep 2: Training the TableGAN model...")
    try:
        start_time = time.time()
        subprocess.run(["python", "paper_exact_implementation.py"], check=True)
        training_time = time.time() - start_time
        print(f"✓ Model training completed successfully in {training_time/60:.2f} minutes")
    except subprocess.CalledProcessError:
        print("✗ Error: Failed to train the model")
        return
    
    # Step 3: Check results
    results_file = 'results/Adult_Full_results.txt'
    if os.path.exists(results_file):
        print("\nStep 3: Results summary")
        with open(results_file, 'r') as f:
            print(f.read())
        print("\nResults are also available in the 'results' directory")
    else:
        print("\nStep 3: Results file not found. Check the 'results' directory for outputs.")
    
    print("\n===== TableGAN Paper Replication Completed =====")
    print("Generated files:")
    print("- Synthetic data: results/Adult_Full_synthetic_data.csv")
    print("- Correlation comparison: results/Adult_Full_correlation_comparison.png")
    print("- Training losses: results/Adult_Full_training_losses.png")
    print("- Full results: results/Adult_Full_results.txt")

if __name__ == "__main__":
    main() 
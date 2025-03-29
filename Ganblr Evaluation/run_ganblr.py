import argparse
import pandas as pd
import logging
from ganblr import GANBLR
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_model(input_file):
    if not os.path.exists(input_file):
        logging.error(f"Dataset file {input_file} does not exist.")
        return

    logging.info(f"Loading dataset from {input_file}...")
    data = pd.read_csv(input_file)

    # Initialize the GANBLR model with the input dimension of the dataset
    model = GANBLR(input_dim=data.shape[1])

    # Train the model
    logging.info("Training the GANBLR model...")
    model.fit(data)

    # Save the trained model
    model.save("ganblr_model_checkpoint.pth")
    logging.info("Training complete. Model saved.")

def generate_data(output_file):
    if not os.path.exists("preprocessed_real_dataset.csv"):
        logging.error("Preprocessed dataset file does not exist.")
        return

    logging.info("Generating synthetic data using GANBLR...")
    data = pd.read_csv("preprocessed_real_dataset.csv")
    model = GANBLR(input_dim=data.shape[1])

    # Load the trained model
    model.load("ganblr_model_checkpoint.pth")
    synthetic_data = model.generate()

    # Save the synthetic data
    synthetic_data.to_csv(output_file, index=False)
    logging.info(f"Synthetic data saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Generate Data with GANBLR")
    parser.add_argument("--train", type=str, help="Path to the training dataset (CSV)")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data")
    parser.add_argument("--output", type=str, default="synthetic_dataset.csv", help="Output file for synthetic data")

    args = parser.parse_args()

    if args.train:
        train_model(args.train)
    if args.generate:
        generate_data(args.output)

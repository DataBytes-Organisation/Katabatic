import argparse
import pandas as pd
import logging
from ganblr import GANBLR

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(input_file):
    try:
        logging.info(f"Loading dataset from {input_file}...")
        data = pd.read_csv(input_file)
        model = GANBLR(input_dim=data.shape[1])
        logging.info("Training the GANBLR model...")
        model.fit(data)
        model.save("ganblr_model_checkpoint.pth")
        logging.info("Training complete. Model saved.")
    except FileNotFoundError as e:
        logging.error(f"File {input_file} not found: {e}")
    except Exception as e:
        logging.error(f"Error during model training: {e}")

def generate_data(output_file):
    try:
        logging.info("Generating synthetic data using GANBLR...")
        data = pd.read_csv("preprocessed_real_dataset.csv")
        model = GANBLR(input_dim=data.shape[1])
        model.load("ganblr_model_checkpoint.pth")
        synthetic_data = model.generate()
        synthetic_data.to_csv(output_file, index=False)
        logging.info(f"Synthetic data saved to {output_file}.")
    except Exception as e:
        logging.error(f"Error during data generation: {e}")

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

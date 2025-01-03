import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Load and display dataset information.")
parser.add_argument("--file", required=True, help="Path to the dataset file")
args = parser.parse_args()

file_path = args.file  # File path provided via command-line argument

try:
    # Load the dataset
    credit_data = pd.read_csv(file_path)

    # Display the first few rows
    print("First few rows of the dataset:")
    print(credit_data.head())

    # Display dataset information
    print("\nDataset Info:")
    credit_data.info()

    # Display summary statistics
    print("\nSummary statistics of numerical columns:")
    print(credit_data.describe())
except FileNotFoundError:
    print(f"Error: The file '{file_path}' does not exist. Please provide a valid file path.")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{file_path}' is empty. Please provide a valid dataset.")
except pd.errors.ParserError as e:
    print(f"Error parsing the file '{file_path}': {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

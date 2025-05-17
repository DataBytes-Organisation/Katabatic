"""
Download and process the UCI Adult dataset as described in the paper:
"Data Synthesis based on Generative Adversarial Networks" by Park et al.

This script prepares the data exactly as described in the paper.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import requests
import io
from zipfile import ZipFile

# Create directories
os.makedirs('data/Adult_Full', exist_ok=True)

# URLs for the Adult dataset
ADULT_TRAIN_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
ADULT_TEST_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

# Column names for the Adult dataset
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

def download_and_prepare_adult():
    """Download the Adult dataset and prepare it as described in the paper."""
    print("Downloading and preparing Adult dataset...")
    
    # Download the Adult training data
    print("Downloading training data...")
    train_response = requests.get(ADULT_TRAIN_URL)
    train_data = pd.read_csv(io.StringIO(train_response.text), 
                            header=None, names=COLUMNS, 
                            sep=', ', engine='python')
    
    # Download the Adult test data
    print("Downloading test data...")
    test_response = requests.get(ADULT_TEST_URL)
    # Remove the first line (header) and fix income column (it has a dot)
    test_text = '\n'.join(test_response.text.split('\n')[1:])
    test_data = pd.read_csv(io.StringIO(test_text), 
                           header=None, names=COLUMNS, 
                           sep=', ', engine='python')
    test_data['income'] = test_data['income'].str.replace('.', '')
    
    # Combine datasets
    print("Combining datasets...")
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Clean data - remove rows with missing values
    print("Cleaning data...")
    combined_data = combined_data.replace('?', np.nan)
    combined_data = combined_data.dropna()
    
    # Process the dataset exactly as described in the paper
    print("Processing data according to paper specifications...")
    
    # 1. Extract labels
    labels = combined_data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    combined_data = combined_data.drop('income', axis=1)
    
    # 2. Encode categorical columns as described in the paper
    categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
                          'relationship', 'race', 'sex', 'native-country']
    
    # The paper uses one-hot encoding in the preprocessing, but for TableGAN input,
    # they use label encoding to convert to numeric format
    for col in categorical_columns:
        le = LabelEncoder()
        combined_data[col] = le.fit_transform(combined_data[col])
    
    # 3. Scale numerical features to [-1, 1] range as per paper
    scaler = MinMaxScaler(feature_range=(-1, 1))
    combined_data = pd.DataFrame(scaler.fit_transform(combined_data), 
                                 columns=combined_data.columns)
    
    # 4. Save the processed dataset
    print("Saving processed dataset...")
    combined_data.to_csv('data/Adult_Full/adult_processed.csv', index=False)
    pd.DataFrame(labels, columns=['income']).to_csv('data/Adult_Full/adult_labels.csv', index=False)
    
    # 5. Save scaler for later use
    np.save('data/Adult_Full/scaler.npy', scaler)
    
    print(f"Dataset processed and saved. Total records: {len(combined_data)}")
    print(f"Distribution of labels: {labels.value_counts()}")
    
    return combined_data, labels, scaler

if __name__ == "__main__":
    download_and_prepare_adult()
    print("Data preparation complete!") 
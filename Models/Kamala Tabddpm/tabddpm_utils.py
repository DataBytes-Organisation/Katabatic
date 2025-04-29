import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_data(data_raw, categorical_threshold=10, datetime_cols=None, drop_cols=None):
    """
    Preprocess the raw data for TabDDPM.
    
    Args:
        data_raw (pd.DataFrame): Raw input data
        categorical_threshold (int): Max unique values to consider a column categorical
        datetime_cols (list): List of datetime column names
        drop_cols (list): List of columns to drop
        
    Returns:
        pd.DataFrame: Preprocessed data
        list: Detected categorical columns
    """
    # Make a copy to avoid modifying the original
    data = data_raw.copy()
    
    # Drop specified columns
    if drop_cols:
        data = data.drop(columns=drop_cols, errors='ignore')
    
    # Detect and convert datetime columns
    if datetime_cols:
        for col in datetime_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
                # Extract useful datetime features
                data[f'{col}_year'] = data[col].dt.year
                data[f'{col}_month'] = data[col].dt.month
                data[f'{col}_day'] = data[col].dt.day
                data[f'{col}_dayofweek'] = data[col].dt.dayofweek
                # Drop the original datetime column
                data = data.drop(columns=[col])
    
    # Handle missing values
    for col in data.columns:
        if data[col].isna().sum() > 0:
            if data[col].dtype in ['int64', 'float64']:
                # Fill numerical missing values with median
                data[col] = data[col].fillna(data[col].median())
            else:
                # Fill categorical missing values with mode
                data[col] = data[col].fillna(data[col].mode()[0])
    
    # Detect and convert categorical columns
    categorical_columns = []
    for col in data.columns:
        n_unique = data[col].nunique()
        # If column has few unique values or is already categorical/object
        if (n_unique <= categorical_threshold and data[col].dtype in ['int64', 'float64']) or \
           data[col].dtype in ['object', 'category']:
            # Convert to categorical type
            data[col] = data[col].astype('category')
            categorical_columns.append(col)
            logging.info(f"Converted {col} to category type (has {n_unique} unique values)")
    
    return data, categorical_columns

def split_data(data, target_column, test_size=0.2, random_state=42):
    """
    Split data into train and test sets.
    
    Args:
        data (pd.DataFrame): The preprocessed data
        target_column (str): The name of the target column
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if y.dtype == 'category' else None
    )
    
    return X_train, X_test, y_train, y_test

def get_tstr_results(evaluation_results):
    """
    Extract Train on Synthetic, Test on Real (TSTR) results from evaluation results.
    
    Args:
        evaluation_results (dict): Dictionary with evaluation results
        
    Returns:
        pd.DataFrame: DataFrame with TSTR performance metrics
    """
    if not evaluation_results or 'tstr_performance' not in evaluation_results:
        return None
    
    tstr_results = evaluation_results['tstr_performance']
    
    # Extract metrics for each model
    metrics_data = {}
    for model_name, metrics in tstr_results.items():
        metrics_data[model_name] = {}
        for metric_name, value in metrics.items():
            if metric_name != 'Error' and value is not None:
                metrics_data[model_name][metric_name] = value
    
    return pd.DataFrame.from_dict(metrics_data, orient='index')
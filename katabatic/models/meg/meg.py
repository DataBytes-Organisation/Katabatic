import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gc
import json
import logging
from concurrent.futures import ProcessPoolExecutor
import yaml

from meg_model import MEGModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    X = np.random.rand(1000, 10)
    y = np.sum(X, axis=1) + np.random.normal(0, 0.1, 1000)
    return X, y

def preprocess_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    try:
        start_memory = get_memory_usage()
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        memory_usage = get_memory_usage() - start_memory

        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'train_time': train_time,
            'inference_time': inference_time,
            'memory_usage': memory_usage
        }
    except Exception as e:
        logging.error(f"Error in train_and_evaluate_model: {str(e)}")
        return None

def test_scalability(model, X, y, sizes=[0.1, 0.25, 0.5, 0.75, 1.0]):
    scalability_results = {}
    for size in sizes:
        n_samples = int(len(X) * size)
        X_subset, y_subset = X[:n_samples], y[:n_samples]
        X_train, X_test, y_train, y_test = preprocess_data(X_subset, y_subset)
        results = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        scalability_results[size] = results
    return scalability_results

def run_benchmarks(config):
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    models = {
        'MEG': MEGModel(),
        'RandomForest': RandomForestRegressor(**config['random_forest']),
        'GradientBoosting': GradientBoostingRegressor(**config['gradient_boosting'])
    }

    results = {}
    scalability_results = {}
    with ProcessPoolExecutor() as executor:
        future_to_model = {executor.submit(train_and_evaluate_model, model, X_train, X_test, y_train, y_test): name for name, model in models.items()}
        for future in future_to_model:
            name = future_to_model[future]
            results[name] = future.result()

        future_to_model = {executor.submit(test_scalability, model, X, y): name for name, model in models.items()}
        for future in future_to_model:
            name = future_to_model[future]
            scalability_results[name] = future.result()

        gc.collect()

    return results, scalability_results

def plot_results(results):
    metrics = ['mse', 'rmse', 'mae', 'r2', 'cv_rmse', 'train_time', 'inference_time', 'memory_usage']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results]
        sns.barplot(x=list(results.keys()), y=values, ax=axes[i])
        axes[i].set_title(metric.upper())
        axes[i].set_ylabel('Value')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    plt.close()

def save_results(results, scalability_results, filename='benchmark_results.json'):
    with open(filename, 'w') as f:
        json.dump({'results': results, 'scalability': scalability_results}, f, indent=2)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    results, scalability_results = run_benchmarks(config)
    plot_results(results)
    save_results(results, scalability_results)

    logging.info("Benchmark Results:")
    for model, metrics in results.items():
        logging.info(f"\n{model}:")
        for metric, value in metrics.items():
            logging.info(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()

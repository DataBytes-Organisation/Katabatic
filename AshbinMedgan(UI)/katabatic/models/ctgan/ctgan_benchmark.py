import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
import logging
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import jensenshannon

# Configure logging to display information-level messages with timestamps
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_likelihood_fitness(real_data, synthetic_data):
    """
    Calculate Likelihood fitness metrics (Lsyn, Ltest) on simulated data.
    
    Lsyn: Log-likelihood of synthetic data under the model trained on synthetic data.
    Ltest: Log-likelihood of real data under the model trained on synthetic data.
    
    Args:
        real_data (pd.DataFrame): The original real dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset generated by CTGAN.
    
    Returns:
        dict: Dictionary containing Lsyn and Ltest values.
    """
    from sklearn.mixture import GaussianMixture

    # Select only numerical columns for likelihood evaluation
    numeric_columns = real_data.select_dtypes(include=['int64', 'float64']).columns
    synthetic_numeric = synthetic_data[numeric_columns].dropna()
    real_numeric = real_data[numeric_columns].dropna()

    # Fit a Gaussian Mixture Model on the synthetic data
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    gmm.fit(synthetic_numeric)

    # Evaluate log-likelihoods
    Lsyn = gmm.score(synthetic_numeric)  # Log-likelihood of synthetic data
    Ltest = gmm.score(real_numeric)      # Log-likelihood of real data under the synthetic model

    return {
        "Lsyn": Lsyn,
        "Ltest": Ltest
    }

def evaluate_statistical_similarity(real_data, synthetic_data):
    """
    Calculate Statistical similarity metrics (Jensen-Shannon Divergence, Wasserstein Distance).
    
    Args:
        real_data (pd.DataFrame): The original real dataset.
        synthetic_data (pd.DataFrame): The synthetic dataset generated by CTGAN.
    
    Returns:
        dict: Dictionary containing mean Jensen-Shannon Divergence and mean Wasserstein Distance.
    """
    js_divergences = []
    wasserstein_distances = []

    # Iterate over each column to compute similarity metrics
    for column in real_data.columns:
        if real_data[column].dtype in ['int64', 'float64']:
            # For numerical columns, compute Wasserstein Distance
            real_values = real_data[column].dropna().values
            synth_values = synthetic_data[column].dropna().values

            wd = wasserstein_distance(real_values, synth_values)
            wasserstein_distances.append(wd)
        else:
            # For categorical columns, compute Jensen-Shannon Divergence
            real_counts = real_data[column].value_counts(normalize=True)
            synth_counts = synthetic_data[column].value_counts(normalize=True)

            # Ensure both distributions have the same categories
            all_categories = set(real_counts.index) | set(synth_counts.index)
            real_probs = real_counts.reindex(all_categories, fill_value=0)
            synth_probs = synth_counts.reindex(all_categories, fill_value=0)

            # Compute Jensen-Shannon Divergence
            js_div = jensenshannon(real_probs, synth_probs)
            js_divergences.append(js_div)

    return {
        "JSD_mean": np.mean(js_divergences) if js_divergences else None,
        "Wasserstein_mean": np.mean(wasserstein_distances) if wasserstein_distances else None
    }

def evaluate_ml_efficacy(X_real, y_real):
    """
    Calculate Machine Learning efficacy metrics (Accuracy, F1, R2) on real data.
    
    Depending on whether the target variable is categorical or numerical, different models are used.
    
    Args:
        X_real (pd.DataFrame): Feature data from the real dataset.
        y_real (pd.Series): Target labels from the real dataset.
    
    Returns:
        dict: Dictionary containing performance metrics for each model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

    # Identify numerical and categorical features for preprocessing
    numeric_features = X_real.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_real.select_dtypes(include=['object', 'category']).columns

    # Define a preprocessing pipeline for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ])

    # Check if the target variable is categorical
    if y_real.dtype == 'object' or y_real.dtype.name == 'category':
        # Define classifiers for categorical targets
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "MLP": MLPClassifier(max_iter=1000),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        results = {}
        for name, clf in classifiers.items():
            # Create a pipeline with preprocessing and the classifier
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])

            # Train the classifier
            pipeline.fit(X_train, y_train)
            # Predict on the test set
            y_pred = pipeline.predict(X_test)

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[name] = {
                "Accuracy": accuracy,
                "F1": f1
            }

    else:
        # Define regressors for numerical targets
        regressors = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "MLP": MLPRegressor(max_iter=1000),
            "XGBoost": XGBRegressor()
        }

        results = {}
        for name, reg in regressors.items():
            # Create a pipeline with preprocessing and the regressor
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', reg)
            ])

            # Train the regressor
            pipeline.fit(X_train, y_train)
            # Predict on the test set
            y_pred = pipeline.predict(X_test)

            # Calculate R-squared score
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                "R2": r2
            }

    return results

def evaluate_tstr(X_real, y_real, X_synthetic, y_synthetic):
    """
    Evaluate Machine Learning utility using the TSTR (Train on Synthetic, Test on Real) approach.
    
    Args:
        X_real (pd.DataFrame): Feature data from the real dataset.
        y_real (pd.Series): Target labels from the real dataset.
        X_synthetic (pd.DataFrame): Feature data from the synthetic dataset.
        y_synthetic (pd.Series): Target labels from the synthetic dataset.
    
    Returns:
        dict: Dictionary containing performance metrics for each model.
    """
    # Identify numerical and categorical features for preprocessing
    numeric_features = X_real.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_real.select_dtypes(include=['object', 'category']).columns

    # Define a preprocessing pipeline for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
        ])

    # Check if the target variable is categorical
    if y_real.dtype == 'object' or y_real.dtype.name == 'category':
        # Define classifiers for categorical targets
        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(),
            "MLP": MLPClassifier(max_iter=1000),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        results = {}
        for name, clf in classifiers.items():
            # Create a pipeline with preprocessing and the classifier
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', clf)
            ])

            # Train the classifier on synthetic data
            pipeline.fit(X_synthetic, y_synthetic)
            # Predict on the real test set
            y_pred = pipeline.predict(X_real)

            # Calculate performance metrics
            accuracy = accuracy_score(y_real, y_pred)
            f1 = f1_score(y_real, y_pred, average='weighted')

            results[name] = {
                "Accuracy": accuracy,
                "F1": f1
            }

    else:
        # Define regressors for numerical targets
        regressors = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(),
            "MLP": MLPRegressor(max_iter=1000),
            "XGBoost": XGBRegressor()
        }

        results = {}
        for name, reg in regressors.items():
            # Create a pipeline with preprocessing and the regressor
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', reg)
            ])

            # Train the regressor on synthetic data
            pipeline.fit(X_synthetic, y_synthetic)
            # Predict on the real test set
            y_pred = pipeline.predict(X_real)

            # Calculate R-squared score
            r2 = r2_score(y_real, y_pred)

            results[name] = {
                "R2": r2
            }

    return results

def evaluate_ctgan(real_data, synthetic_data):
    """
    Evaluate the CTGAN model using various metrics including likelihood fitness,
    statistical similarity, machine learning efficacy, and TSTR performance.
    
    Args:
        real_data (pd.DataFrame): The original real dataset, including the target column named 'Category'.
        synthetic_data (pd.DataFrame): The synthetic dataset generated by CTGAN, including the target column named 'Category'.
    
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    # Separate features and target variable from real data
    X_real = real_data.drop('Category', axis=1)
    y_real = real_data['Category']
    
    # Separate features and target variable from synthetic data
    X_synthetic = synthetic_data.drop('Category', axis=1)
    y_synthetic = synthetic_data['Category']
    
    # Handle any missing values in the synthetic target variable by imputing the mode
    y_synthetic = y_synthetic.fillna(y_synthetic.mode().iloc[0])
    
    results = {}

    # Evaluate likelihood fitness metrics
    try:
        results["likelihood_fitness"] = evaluate_likelihood_fitness(X_real, X_synthetic)
    except Exception as e:
        print(f"Error in likelihood_fitness: {str(e)}")
        results["likelihood_fitness"] = None

    # Evaluate statistical similarity metrics
    try:
        results["statistical_similarity"] = evaluate_statistical_similarity(X_real, X_synthetic)
    except Exception as e:
        print(f"Error in statistical_similarity: {str(e)}")
        results["statistical_similarity"] = None

    # Evaluate machine learning efficacy on real data
    try:
        results["ml_efficacy"] = evaluate_ml_efficacy(X_real, y_real)
    except Exception as e:
        print(f"Error in ml_efficacy: {str(e)}")
        results["ml_efficacy"] = None

    # Evaluate TSTR performance (train on synthetic, test on real)
    try:
        results["tstr_performance"] = evaluate_tstr(X_real, y_real, X_synthetic, y_synthetic)
    except Exception as e:
        print(f"Error in tstr_performance: {str(e)}")
        results["tstr_performance"] = None

    return results

def print_evaluation_results(results):
    """
    Print the evaluation results in a structured and readable format using logging.
    
    Args:
        results (dict): Dictionary containing all evaluation metrics.
    """
    logging.info("CTGAN Evaluation Results:")

    # Print Likelihood Fitness Metrics
    if results.get("likelihood_fitness") is not None:
        logging.info("\nLikelihood Fitness Metrics:")
        logging.info(f"  - Lsyn (Synthetic Data Log-Likelihood): {results['likelihood_fitness']['Lsyn']:.4f}")
        logging.info(f"  - Ltest (Real Data Log-Likelihood under Synthetic Model): {results['likelihood_fitness']['Ltest']:.4f}")
    else:
        logging.info("Likelihood Fitness: Error occurred during calculation")

    # Print Statistical Similarity Metrics
    if results.get("statistical_similarity") is not None:
        logging.info("\nStatistical Similarity Metrics:")
        if results['statistical_similarity']['JSD_mean'] is not None:
            logging.info(f"  - Jensen-Shannon Divergence Mean (Categorical): {results['statistical_similarity']['JSD_mean']:.4f}")
        if results['statistical_similarity']['Wasserstein_mean'] is not None:
            logging.info(f"  - Wasserstein Distance Mean (Numerical): {results['statistical_similarity']['Wasserstein_mean']:.4f}")
    else:
        logging.info("\nStatistical Similarity: Error occurred during calculation")

    # Print Machine Learning Efficacy Metrics
    if results.get("ml_efficacy") is not None:
        logging.info("\nMachine Learning Efficacy Metrics on Real Data:")
        for model_name, metrics in results['ml_efficacy'].items():
            logging.info(f"  {model_name}:")
            for metric_name, value in metrics.items():
                logging.info(f"    - {metric_name}: {value:.4f}")
    else:
        logging.info("\nMachine Learning Efficacy: Error occurred during calculation")

    # Print TSTR Performance Metrics
    if results.get("tstr_performance") is not None:
        logging.info("\nMachine Learning Utility using TSTR Approach:")
        for model_name, metrics in results['tstr_performance'].items():
            logging.info(f"  {model_name}:")
            for metric_name, value in metrics.items():
                logging.info(f"    - {metric_name}: {value:.4f}")
    else:
        logging.info("\nTSTR Performance: Error occurred during calculation")

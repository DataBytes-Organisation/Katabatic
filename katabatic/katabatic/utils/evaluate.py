import os
import pandas as pd
import json
from ..importer import *  # Aiko Services module loader
import logging

METRICS_FILE = os.path.abspath(
    "katabatic/metrics/metrics.json"
)  # Constant to retrieve metrics function table


# Accepts metric_name:str. Returns an instance of the selected metric.
# TODO: possibly update METRICS_FILE to a dict of dicts (to include type etc.. of each metric)
def run_metric(metric_name):
    with open(METRICS_FILE, "r") as file:
        metrics = json.load(file)

        if not metric_name in metrics:
            raise SystemExit(
                f"Metrics Function Table '{METRICS_FILE}' doesn't contain metric: {metric_name}"
            )
        metric = metrics[metric_name]

    diagnostic = None  # initialise an empty diagnostic variable
    try:
        module = load_module(metric)  # load_module method from Aiko services
    except FileNotFoundError:
        diagnostic = "could not be found."
    except Exception as exception:
        diagnostic = f"could not be loaded: {exception}"
    if diagnostic:
        raise SystemExit(f"Metric {metric_name} {diagnostic}")
    # Run Metric
    # result = metric_name.evaluate()
    # return result
    return module


# evaluate_data assumes the last column to be y and all others to be X
def evaluate_data(synthetic_data, real_data, data_type, dict_of_metrics):
    """
    Evaluate the quality of synthetic data against real data using specified metrics.

    This function assumes that the last column of the data is the target variable (y),
    and all other columns are features (X). It then evaluates the performance of
    the synthetic data using various metrics provided in `dict_of_metrics`.

    Parameters:
    - synthetic_data (pd.DataFrame): The synthetic dataset to evaluate.
    - real_data (pd.DataFrame): The real dataset to compare against.
    - data_type (str): The type of data, either 'discrete' or 'continuous'.
    - dict_of_metrics (dict): A dictionary where keys are metric names and values are
                              metric functions or classes that have an `evaluate` method.

    Returns:
    - pd.DataFrame: A DataFrame with the metric names and their corresponding evaluation values.
    """

    # Input Validation
    if not isinstance(synthetic_data, pd.DataFrame) or not isinstance(
        real_data, pd.DataFrame
    ):
        raise ValueError("Both synthetic_data and real_data must be pandas DataFrames.")

    if synthetic_data.shape != real_data.shape:
        logging.warning(
            "Input shapes do not match: synthetic_data shape: %s, real_data shape: %s",
            synthetic_data.shape,
            real_data.shape,
        )

    # Reset Column Headers for both datasets
    synthetic_data.columns = range(synthetic_data.shape[1])
    real_data.columns = range(real_data.shape[1])

    # Split X and y, assume y is the last column.
    X_synthetic, y_synthetic = synthetic_data.iloc[:, :-1], synthetic_data.iloc[:, -1]
    X_real, y_real = real_data.iloc[:, :-1], real_data.iloc[:, -1]

    # Initialize the results DataFrame
    results_df = pd.DataFrame(columns=["Metric", "Value"])

    # Evaluate each metric
    for metric_name in dict_of_metrics:
        try:
            # Evaluate the metric
            metric_module = run_metric(metric_name)
            result = metric_module.evaluate(X_synthetic, y_synthetic, X_real, y_real)
            logging.info("Successfully evaluated metric: %s", metric_name)
        except Exception as e:
            logging.error("Error evaluating metric %s: %s", metric_name, str(e))
            result = None

        # Append the result to the results DataFrame
        results_df = pd.concat(
            [results_df, pd.DataFrame({"Metric": [metric_name], "Value": [result]})],
            ignore_index=True,
        )

    return results_df


def evaluate_models(real_data, dict_of_models, dict_of_metrics):
    results_df = pd.DataFrame()
    for model_name, model in dict_of_models.items():
        model_results = evaluate_data(
            real_data,
            pd.DataFrame(model.generate(size=len(real_data))),
            "continuous",
            dict_of_metrics,
        )
        model_results["Model"] = model_name
        results_df = pd.concat([results_df, model_results], ignore_index=True)

    # results_df = pd.DataFrame()
    # for i in range(len(dict_of_models)):
    #     model_name = dict_of_models[i]

    # run_model
    return results_df

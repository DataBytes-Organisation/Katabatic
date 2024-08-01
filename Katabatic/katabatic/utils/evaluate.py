import os
import pandas as pd
import json
from importer import load_module   # Aiko Services module loader

METRICS_FILE = os.path.abspath("metrics/metrics.json")  # Constant to retrieve metrics function table


# Accepts metric_name:str. Returns an instance of the selected metric.
# TODO: possibly update METRICS_FILE to a dict of dicts (to include type etc.. of each metric)
def run_metric(metric_name):

    with open(METRICS_FILE, "r") as file:
        metrics = json.load(file)

        if not metric_name in metrics:
            raise SystemExit(
                f"Metrics Function Table '{METRICS_FILE}' doesn't contain metric: {metric_name}")
        metric = metrics[metric_name]

    diagnostic = None # initialise an empty diagnostic variable  
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
def evaluate_data(synthetic_data, real_data, data_type, dict_of_metrics):   #data_type s/be either 'discrete' or 'continuous'
    
    # Check if synthetic_data and real_data are uniform in type, shape and columns
    if not type(synthetic_data)==type(real_data):
        print("WARNING: Input types do not match: synthetic_data type: ", type(synthetic_data),"real_data type: ", type(real_data))
    if not synthetic_data.shape==real_data.shape:
        print("WARNING: Input shapes do not match: synthetic_data shape: ", synthetic_data.shape,"real_data shape: ", real_data.shape)
    # if not synthetic_data.columns.all()==real_data.columns.all():
    #     print("WARNING: Input column headers do not match: synthetic_data headers: ", synthetic_data.columns,"real_data headers: ", real_data.columns)

    # Reset Column Headers for both datasets
    synthetic_data.columns = range(synthetic_data.shape[1])
    real_data.columns = range(real_data.shape[1])

    # Split X and y, assume y is the last column.
    X_synthetic, y_synthetic = synthetic_data.iloc[:,:-1], synthetic_data.iloc[:,-1:]
    X_real, y_real = real_data.iloc[:,:-1], real_data.iloc[:,-1:]

    results_df = pd.DataFrame({"Metric": [], "Value": []})
    # By default use TSTR with Logistic Regression for discrete models
    for key in dict_of_metrics:
        metric_module = run_metric(key)
        result = metric_module.evaluate(X_synthetic, y_synthetic, X_real, y_real)    # TODO: update parameters of the evaluate function so they work for every metric.
        new_row = pd.DataFrame({"Metric": [key], "Value": [result]})
        results_df = pd.concat([results_df, new_row], ignore_index = True)
        #function = METRICS_FILE.key.value

    return results_df

def evaluate_models(real_data, dict_of_models, dict_of_metrics):

    results_df = pd.DataFrame()
    for i in range(len(dict_of_models)):
        model_name = dict_of_models[i]
        
    #run_model
    return


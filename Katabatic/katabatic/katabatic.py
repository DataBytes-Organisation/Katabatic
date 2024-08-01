# This file declaratively loads the configured module and instantiates the Tabular Data Generative Model (TDGM) class.
import os
import sys
import json
import pandas as pd
from multiprocessing import Process
from katabatic_spi import KatabaticModelSPI  # Katabatic Model SPI 
from importer import load_module   # Aiko Services module loader
# from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings("ignore")
import logging
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
logging.basicConfig(filename='running.log', level=logging.INFO)

CONFIG_FILE = os.path.abspath("katabatic_config.json")  # Constant to retrieve config file
METRICS_FILE = os.path.abspath("metrics/metrics.json")  # Constant to retrieve metrics function table

class Katabatic():

    def run_model(model_name):
        print(f"--------------------------")
        print(f"module name:    {__name__}")
        print(f"parent process: {os.getppid()}")
        print(f"process id:     {os.getpid()}")

        with open(CONFIG_FILE, "r") as file: # Open config file as read only
            config = json.load(file) # config is a dict of dicts, each containing config variables for a model.

            if not model_name in config:
                raise SystemExit(
                    f"Configuration '{CONFIG_FILE}' doesn't have model: {model_name}")
            config = config[model_name] # update config to just one dict, containing config variables of a single model.

        try:
            module_name = config["tdgm_module_name"]
            class_name = config["tdgm_class_name"]
        except KeyError as key_error:
            raise SystemExit(   # SystemExit exception prints the below and immediately exits the interpreter
                f"Configuration file '{CONFIG_FILE}' does not contain: {key_error}")
        
        diagnostic = None # initialise an empty diagnostic variable
        try:
            # breakpoint()
            print(module_name)
            module = load_module(module_name)  # load_module method from Aiko services
            model_class = getattr(module, class_name)
        except FileNotFoundError:
            diagnostic = "could not be found."
        except Exception as exception:
            diagnostic = f"could not be loaded: {exception}"
        if diagnostic:
            raise SystemExit(f"Module {module_name} {diagnostic}")

        model = model_class() # Create an instance of the model class
        if not isinstance(model, KatabaticModelSPI):
            raise SystemExit(f"{class_name} doesn't implement KatabaticModelSPI.")

        return model
        # # TODO: move the next code block outside the load_model function
        # model.load_model()
        # model.fit(X_train, y_train)
        # synthetic_data = pd.DataFrame(model.generate())
        # return synthetic_data

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
    # evaluate_data evaluates the synthetic data based on the training dataset for a variety of metrics. 
    # TODO: Update third parameter to something like discrete_or_continuous or discrete (True or False)
    def evaluate_data(synthetic_data, real_data, data_type, dict_of_metrics):   #data_type s/be either 'discrete' or 'continuous'
        # Convert column headers to integers?
        # Remove column headers?
        # print("Type of real_data: ", type(real_data))
        # print("Type of synthetic_data: ", type(synthetic_data))

        # Check if synthetic_data and real_data are uniform in type, shape and columns
        if not type(synthetic_data)==type(real_data):
            raise SystemExit("WARNING: Input types do not match: synthetic_data type: ", type(synthetic_data),"real_data type: ", type(real_data))
        if not synthetic_data.shape==real_data.shape:
            raise SystemExit("WARNING: Input shapes do not match: synthetic_data shape: ", synthetic_data.shape,"real_data shape: ", real_data.shape)
        if not synthetic_data.columns.all()==real_data.columns.all():
            raise SystemExit("WARNING: Input column headers do not match: synthetic_data headers: ", synthetic_data.columns,"real_data headers: ", real_data.columns)

        # Reset Column Headers for both datasets
        synthetic_data.columns = range(synthetic_data.shape[1])
        real_data.columns = range(real_data.shape[1])

        # Split X and y, assume y is the last column.
        X_synthetic, y_synthetic = synthetic_data.iloc[:,:-1], synthetic_data.iloc[:,-1:]
        X_real, y_real = real_data.iloc[:,:-1], real_data.iloc[:,-1:]

        results_df = pd.DataFrame({"Metric": [], "Value": []})
        # By default use TSTR with Logistic Regression for discrete models
        for key in dict_of_metrics:
            metric_module = Katabatic.run_metric(key)
            result = metric_module.evaluate(X_synthetic, y_synthetic, X_real, y_real)    # TODO: update parameters of the evaluate function so they work for every metric.
            new_row = pd.DataFrame({"Metric": [key], "Value": [result]})
            results_df = pd.concat([results_df, new_row], ignore_index = True)
            #function = METRICS_FILE.key.value

        return results_df

    # TODO: return a dynamic comparison report of 
    def evaluate_models(real_data, dict_of_models, dict_of_metrics):

        results_df = pd.DataFrame()
        for i in range(len(dict_of_models)):
            model_name = dict_of_models[i]

        #run_model
        return

    # Returns X_train, X_test, y_train, y_test in that order
    def preprocessing(data):

        # Reset Column Headers 
        data.columns = range(data.shape[1])
        
        # Handle null values
        for col in range(len(data.columns)):     # Check which features contain null values
            if data[col].isnull().any():
                print("Column: ", col, " contains Null values.")


        # data=data.copy().fillna(data['f21'].median()) #copy the dataframe into a new dataframe, and fill missing values. 
        # Validate Data

        # X, Y Split
        X = data.iloc[:, :-1] # all but last column
        y = data.iloc[:, -1] # last column
        # print("Shape of X: ", X.shape, "Shape of y: ", y.shape)

        # Train, Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        print("Shape of X_train: ", X_train.shape, "Shape of y_train: ", y_train.shape)
        print("Shape of X_test: ", X_test.shape, "Shape of y_test: ", y_test.shape)
        print("Type of X_test: ", type(X_test), "Type of y_test: ", type(y_test))


        return X_train, X_test, y_train, y_test
        

if __name__ == "__main__":
    print(f"[Welcome to Katabatic version 0.1]")
    print(f"module name:    {__name__}")
    print(f"parent process: {os.getppid()}")
    print(f"process id:     {os.getpid()}")
    import warnings
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 2:
        raise SystemExit("Usage: katabatic.py MODEL_NAME ...")
    arguments = sys.argv[1:]
    
    for index in range(len(arguments)):

        model_name = arguments[index]  # Accept the argument as model_name
        model = Katabatic.run_model(model_name)  # Create an instance of the specified model

    # Get demo data from GANBLR package
    from pandas import read_csv
    ganblr_demo_data = read_csv('https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv',dtype=int)

    X_train, X_test, y_train, y_test = Katabatic.preprocessing(ganblr_demo_data)

    
                # demo_data = pd.read_csv('cities_demo.csv') # Retrieve some demo data
                # X_train, y_train = demo_data[["Temperature","Latitude","Longitude"]], demo_data["Category"] # Split X from y
   
   # TODO: Add a module for generating demo data.  
    # X_train, y_train = load_breast_cancer(return_X_y=True, as_frame=True)
    # demo_data = load_breast_cancer()
    # model.load_data(demo_data)
    
    model.load_model()
    print("1 Shape of y_train: ", y_train.shape)
    model.fit(X_train, y_train) # Fit the model to the data # Louka q: Do I really need to pass y_train ?
    print("---- SHOW REAL DATA ----")
    real_data = pd.concat([X_train, y_train], axis=1)
    print(real_data.head())

    print("--- GENERATE SYNTHETIC DATA ---")   
    synthetic_data = pd.DataFrame(model.generate()) # Generate synthetic data
    synthetic_data.to_csv("output.csv")  # Save output to csv

    print(synthetic_data.head())    # Show a sample of the synthetic data output

    print("--- EVALUATE SYNTHETIC DATA ---")   # Evaluate the Synthetic Data
    real_data = pd.concat([X_test, y_test],axis=1)
    data_eval_result = Katabatic.evaluate_data(synthetic_data, real_data, "discrete",{'tstr_logreg'})   # Evaluate the synthetic data and show the result
    
    print(data_eval_result)
    
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import time
import numpy as np
from sklearn import preprocessing
from ..ctabganp.pipeline.data_preparation import DataPrep
from ..ctabganp.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

class CTABGANP():

    def __init__(self,
                 raw_csv_path = "Real_Datasets/Adult.csv",
                 headers = None,
                 sep = None,
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 general_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": "income"},
                 preprocessing_raw_df = None):

        self.__name__ = 'CTABGANP'
              
        self.synthesizer = CTABGANSynthesizer()
        if headers is not None:
            if sep is not None:
                self.raw_df = pd.read_csv(raw_csv_path, sep=sep, names=headers)
            else:
                self.raw_df = pd.read_csv(raw_csv_path, names=headers)
        else:
            self.raw_df = pd.read_csv(raw_csv_path)

        if preprocessing_raw_df is not None:
            self.raw_df = preprocessing_raw_df(self.raw_df)

        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.x_train = self.data_prep.x_train
        self.y_train = self.data_prep.y_train
        self.x_test = self.data_prep.x_test
        self.y_test = self.data_prep.y_test

    def fit(self, epochs=100):
        start_time = time.time()
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type, epochs=epochs)
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self, size=None):
        
        sample = self.synthesizer.sample(len(self.raw_df), size) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import numpy as np
from .tablegan import TableGAN
from .tablegan_utils import preprocess_data, postprocess_data

class TableGANAdapter(KatabaticModelSPI):
    
    def __init__(self, type='continuous', privacy_setting='low'):
        self.type = type
        self.privacy_setting = privacy_setting
        self.constraints = None 
        self.batch_size = 64
        self.epochs = 100
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.input_dim = None
        self.label_dim = None
        self.training_sample_size = 0

    def load_model(self):
        print("---Initialise TableGAN Model")
        if self.input_dim is None or self.label_dim is None:
            raise ValueError("input_dim and label_dim must be set before loading the model")
        
        # Set privacy parameters based on privacy_setting
        if self.privacy_setting == 'low':
            delta_mean = 0.0
            delta_sd = 0.0
        elif self.privacy_setting == 'high':
            delta_mean = 0.2
            delta_sd = 0.2
        else:
            delta_mean = 0.1
            delta_sd = 0.1

        self.model = TableGAN(input_dim=self.input_dim, label_dim=self.label_dim, 
                              delta_mean=delta_mean, delta_sd=delta_sd)
        return self.model


    def load_data(self, data_pathname):
        print("Loading Data...")
        try:
            data = pd.read_csv(data_pathname)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def fit(self, X_train, y_train, epochs=None, batch_size=None):
        print(f"---FIT TableGAN Model with {self.privacy_setting} privacy setting")
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size

        X_processed, y_processed, self.scaler, self.label_encoder = preprocess_data(X_train, y_train)
        self.input_dim = X_processed.shape[1] * X_processed.shape[2]
        self.label_dim = y_processed.shape[1]

        if self.model is None:
            self.load_model()

        self.model.fit(X_processed, y_processed, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        self.training_sample_size = len(X_train)

    def generate(self, size=None):
        print("---Generate from TableGAN Model")
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() before generate().")

        if size is None:
            size = self.training_sample_size

        generated_data, generated_labels = self.model.sample(size)
        generated_data = postprocess_data(generated_data, generated_labels, self.scaler, self.label_encoder)
        return generated_data
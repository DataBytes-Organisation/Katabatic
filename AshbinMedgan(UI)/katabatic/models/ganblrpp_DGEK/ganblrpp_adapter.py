
from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import numpy as np
from .ganblrpp import GANBLRPP 

from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import numpy as np
from .ganblrpp import GANBLRPP 

class GanblrppAdapter(KatabaticModelSPI):
    def __init__(self, model_type="discrete", numerical_columns=None, random_state=None):
        self.type = model_type
        self.model = None
        self.constraints = None
        self.batch_size = None
        self.epochs = None
        self.training_sample_size = 0
        self.numerical_columns = numerical_columns
        self.random_state = random_state

    def load_model(self):
        if self.numerical_columns is None:
            raise ValueError("Numerical columns must be provided for GANBLRPP initialization.")
        
        print("[INFO] Initializing GANBLR++ Model")
        self.model = GANBLRPP(numerical_columns=self.numerical_columns, random_state=self.random_state)
        
        if self.model is None:
            raise RuntimeError("Failed to initialize GANBLR++ model.")
        
        return self.model

    def load_data(self, data_pathname):
        print(f"[INFO] Loading data from {data_pathname}")
        try:
            data = pd.read_csv(data_pathname)
            print("[SUCCESS] Data loaded successfully.")
        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            raise
        return data

    def fit(self, X_train, y_train, k=0, epochs=10, batch_size=64):
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call `load_model()` first.")
        
        try:
            print("[INFO] Training GANBLR++ model")
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.values
            if isinstance(y_train, pd.Series):
                y_train = y_train.values
            
            self.model.fit(X_train, y_train, k=k, batch_size=batch_size, epochs=epochs, verbose=0)
            self.training_sample_size = len(X_train)
            print("[SUCCESS] Model training completed")
        except Exception as e:
            print(f"[ERROR] An error occurred during model training: {e}")
            raise

    def generate(self, size=None):
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call `load_model()` first.")
        
        try:
            print("[INFO] Generating data using GANBLR++ model")
            if size is None:
                size = self.training_sample_size

            generated_data = self.model.sample(size, verbose=0)
            if isinstance(generated_data, np.ndarray):
                generated_data = pd.DataFrame(generated_data)
                
            print("[SUCCESS] Data generation completed")
            return generated_data
        except Exception as e:
            print(f"[ERROR] An error occurred during data generation: {e}")
            raise
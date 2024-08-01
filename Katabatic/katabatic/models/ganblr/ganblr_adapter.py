from katabatic_spi import KatabaticModelSPI
import pandas as pd
from .ganblr import GANBLR

class GanblrAdapter(KatabaticModelSPI):
    
    def __init__(self, type='discrete'):
        self.type = type  # Should be either 'discrete' or 'continuous'
        self.constraints = None 
        self.batch_size = None
        self.epochs = None

    def load_model(self):
        print("---Initialise Ganblr Model")
        self.model = GANBLR() # Initialise and return an instance of the ganblr model. 
        self.training_sample_size = 0
        return self.model

    #TODO: add exception handling to load()
    def load_data(self, data_pathname):
        data = pd.DataFrame
        print("Loading Data...")
        return data
    
    #TODO: add exception handling to fit()
    def fit(self, X_train, y_train, k=0, epochs=10, batch_size=64):   #TODO: remove hard coded numbers
        print("---FIT Ganblr Model")
        self.model.fit(X_train, y_train, k, batch_size=batch_size, epochs=epochs, verbose=0) # TODO: train self.model on the input data
        self.training_sample_size = len(X_train)
        return   # Don't need to return anything. 

    #TODO: add exception handling to generate()
    def generate(self, size=None):  #Modify so that if size is not specified, default is the training sample size.
        print("---Generate from Ganblr Model")
        if size==None: 
            size = self.training_sample_size

        generated_data = self.model.sample(size, verbose=0)
        return generated_data
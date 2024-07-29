"""Abstract Base class for all models."""

# Author: Jaime Blackwell
# License: GNU Affero License  
import abc
from abc import ABC, abstractmethod   # Import abstraction Module

class Model(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):

        '''
                    *** MODEL CLASS PARAMETERS ***
        ------------------------------------------------------------
        weights : initial weights for data generation. Default = None
        epochs : number of epochs to train the model. Default = None
        batch_size : batch size for training. Default = None
        self.x : Training set for fitting to the model
        self.Y : 
        -------------------------------------------------------------
        '''
        self.weights = None
        self.epochs = None
        self.batch_size = None
        self.x = None
        self.Y = None
        self.k = 0

    @abstractmethod
    def fit(self, x, Y):    # sklearn-style fit method   ,x, Y, **kwargs
        self.x = x
        self.Y = Y
        return NotImplementedError('model.fit() must be defined in the concrete class')

    @abstractmethod
    def generate(self):      # , size=None, **kwargs
        return NotImplementedError('model.generate() must be defined in the concrete class')

    @abstractmethod
    def evaluate(self):   # x, Y, classifier
        pass
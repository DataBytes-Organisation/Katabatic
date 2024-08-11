from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.weights = None
        self.epochs = None
        self.batch_size = None
        self.x = None
        self.Y = None
        self.k = 0

    @abstractmethod
    def fit(self, x, Y):
        self.x = x
        self.Y = Y
        return NotImplementedError('model.fit() must be defined in the concrete class')

    @abstractmethod
    def generate(self):
        return NotImplementedError('model.generate() must be defined in the concrete class')

    @abstractmethod
    def evaluate(self):
        pass

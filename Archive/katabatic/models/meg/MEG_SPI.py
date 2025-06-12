from MEG_SPI import MEGModelSPI

class MEGModelSPI(ABC):
    
    @abstractmethod
    def __init__(self, model_type):
        self.model_type = model_type  # The type of model, e.g., 'ensemble', 'masked', etc.
        self.parameters = None  # Placeholder for model parameters
        self.masking_strategy = None  # Placeholder for the masking strategy to be used
        self.num_models = None  # The number of models in the ensemble

    @abstractmethod
    def load_model(self): 

        pass

    @abstractmethod
    def load_data(self, data):

        pass

    @abstractmethod
    def train(self):  

        pass

    @abstractmethod
    def generate(self, num_samples):

        pass

class MEGMetricSPI(ABC):
    
    @abstractmethod
    def evaluate(self, real_data, synthetic_data): 

        pass

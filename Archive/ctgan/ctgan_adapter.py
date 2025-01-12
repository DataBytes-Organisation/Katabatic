from katabatic.katabatic_spi import KatabaticModelSPI
import pandas as pd
import sdv
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


class CtganAdapter(KatabaticModelSPI):

    def __init__(self, type="continuous"):
        self.type = None  # Should be either 'discrete' or 'continuous'
        self.constraints = None
        self.batch_size = None
        self.epochs = None
        self.data = None

    #     #TODO: add exception handling to load()
    def load_model(self):  # Load the model
        metadata = SingleTableMetadata()
        metadata.detect_from_csv(
            filepath="/Users/abdullah/Documents/GitHub/Katabatic/katabatic/nursery/nursery.data"
        )
        data = pd.read_csv(
            "/Users/abdullah/Documents/GitHub/Katabatic/katabatic/nursery/nursery.data"
        )
        self.load_data(data)
        self.model = CTGANSynthesizer(
            metadata
        )  # Initialise and return an instance of the model
        # print("Loading the model")
        return

    def load_data(self, data):  # Load data
        self.data = data

    #     #TODO: add exception handling to fit()
    def fit(self, X_train, y_train, k=0, epochs=10, batch_size=64):  # Fit model to data
        self.model.fit(self.data)
        print("Fitting the model")
        return

    #     TODO: add exception handling to generate()
    def generate(self, size=13):  # Generate synthetic data
        return self.model.sample(num_rows=size)

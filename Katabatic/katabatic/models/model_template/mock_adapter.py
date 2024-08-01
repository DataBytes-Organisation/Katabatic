from katabatic_spi import KatabaticModelSPI

class MockAdapter(KatabaticModelSPI):

    def __init__(self, type='discrete'):
        self.type = None  # Should be either 'discrete' or 'continuous'
        self.constraints = None 
        self.batch_size = None
        self.epochs = None

    def load_model(self):
        print('Loading the model')

    def load_data(self, data_pathname):
        print("Loading data now.")
    
    def fit(self, X_train, y_train, k=0, epochs=10, batch_size=64):
        print("Fitting the model now.")

    def generate(self, size=None): 
        print("Generating data now.")
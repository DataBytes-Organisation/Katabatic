# A dummy datagen model written by Jaime Blackwell

from katabatic.models.model import Model   # Import the abstract base class

# This dummy model simply duplicates the data and returns it. 
# Adding an extra comment for no reason
class DummyModel(Model):

    def __init__(self, x, Y, batch_size=64):
        super().__init__("model")
        self.batch_size = batch_size
        self.x = x   # data to train on
        self.Y = Y   # Y is the target variable
        self.k = 0   

    def fit(self, x, Y):  
        self.x = x
        self.Y = Y
        return 42

    def generate(self):
        return 42

    def evaluate(self):
        return 42
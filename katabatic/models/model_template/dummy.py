from model import Model
import unittest

class DummyModel(Model):
    def __init__(self, x, Y, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.x = x
        self.Y = Y
        self.k = 0

    def fit(self, x, Y):
        self.x = x
        self.Y = Y
        return 42

    def generate(self):
        return 42

    def evaluate(self):
        return 42

def test_dummy_model():
    model = DummyModel(None, None)
    assert model.fit(None, None) == 42
    assert model.generate() == 42
    assert model.evaluate() == 42
    print("All tests passed!")

if __name__ == "__main__":
    test_dummy_model()

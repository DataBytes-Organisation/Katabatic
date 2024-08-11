import pytest
import numpy as np
from ganblr import GANBLR

@pytest.fixture
def sample_data():
    X = np.random.randint(0, 5, size=(100, 10))
    y = np.random.randint(0, 2, size=100)
    return X, y

def test_ganblr_fit(sample_data):
    X, y = sample_data
    model = GANBLR()
    model.fit(X, y, epochs=1)
    assert model._d is not None

def test_ganblr_sample(sample_data):
    X, y = sample_data
    model = GANBLR()
    model.fit(X, y, epochs=1)
    samples = model.sample(10)
    assert samples.shape == (10, X.shape[1] + 1)

def test_ganblr_evaluate(sample_data):
    X, y = sample_data
    model = GANBLR()
    model.fit(X, y, epochs=1)
    score = model.evaluate(X, y)
    assert 0 <= score <= 1

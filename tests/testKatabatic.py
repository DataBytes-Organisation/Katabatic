#from katabatic.katabatic import Katabatic   #TODO: ideally simplify the import statement
from katabatic import *
from sklearn import datasets
import numpy as np

# Test #1 involves generating some synthetic data using the Katabatic framework
# To do that we will gather demo data from scikit-learn...
iris = datasets.load_iris()
categories = iris.target_names
print("Categories: ", categories)
print("Feature Names: ", iris.feature_names)

data = iris.data
target = iris.target

# First instantiate a model:

#model1 = Katabatic.run_model('ganblr')

#model1.fit(data, target)

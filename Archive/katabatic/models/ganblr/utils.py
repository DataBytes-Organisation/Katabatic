from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

class softmax_weight(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be under softmax `."""
  
    def __init__(self,feature_uniques):
        if isinstance(feature_uniques, np.ndarray):
            idxs = math_ops.cumsum(np.hstack([np.array([0]),feature_uniques]))
        else:
            idxs = math_ops.cumsum([0] + feature_uniques)
        idxs = [i.numpy() for i in idxs]
        self.feature_idxs = [
            (idxs[i],idxs[i+1]) for i in range(len(idxs)-1)
        ]
  
    def __call__(self, w):     
        w_new = [
            math_ops.log(tf.nn.softmax(w[i:j,:], axis=0))
            for i,j in self.feature_idxs
        ]
        return tf.concat(w_new, 0)
  
    def get_config(self):
        return {'feature_idxs': self.feature_idxs}

def elr_loss(KL_LOSS):
  def loss(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)+ KL_LOSS
  return loss

def KL_loss(prob_fake):
    return np.mean(-np.log(np.subtract(1,prob_fake)))

def get_lr(input_dim, output_dim, constraint=None,KL_LOSS=0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(output_dim, input_dim=input_dim, activation='softmax',kernel_constraint=constraint))
    model.compile(loss=elr_loss(KL_LOSS), optimizer='adam', metrics=['accuracy'])
    #log_elr = model.fit(*train_data, validation_data=test_data, batch_size=batch_size,epochs=epochs)
    return model 

def sample(*arrays, n=None, frac=None, random_state=None):
    '''
    generate sample random arrays from given arrays. The given arrays must be same size.
    
    Parameters:
    --------------
    *arrays: arrays to be sampled.

    n (int): Number of random samples to generate.

    frac: Float value between 0 and 1, Returns (float value * length of given arrays). frac cannot be used with n.

    random_state: int value or numpy.random.RandomState, optional. if set to a particular integer, will return same samples in every iteration.

    Return:
    --------------
    the sampled array(s). Passing in multiple arrays will result in the return of a tuple.

    '''
    random = np.random
    if isinstance(random_state, int):
        random = random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        random = random_state
    
    arr0 = arrays[0]
    original_size = len(arr0)
    if n == None and frac == None:
        raise Exception('You must specify one of frac or size.')
    if n == None:
        n = int(len(arr0) * frac)

    idxs = random.choice(original_size, n, replace=False)
    if len(arrays) > 1:
        sampled_arrays = []
        for arr in arrays:
            assert(len(arr) == original_size)
            sampled_arrays.append(arr[idxs])
        return tuple(sampled_arrays)
    else:
        return arr0[idxs]

DEMO_DATASETS = {
    'adult': {
        'link':'https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv',
        'params': {
            'dtype' : int
        }
    },
    'adult-raw':{
        'link':'https://drive.google.com/uc?export=download&id=1iA-_qIC1xKQJ4nL2ugX1_XJQf8__xOY0',
        'params': {}
    }
}

def get_demo_data(name='adult'):
    """
    Download demo dataset from internet.

    Parameters
    ----------
    name : str 
        Name of dataset. Should be one of ['adult', 'adult-raw'].

    Returns
    -------
    data : pandas.DataFrame
        the demo dataset.
    """
    assert(name in DEMO_DATASETS.keys())
    return read_csv(DEMO_DATASETS[name]['link'], **DEMO_DATASETS[name]['params'])

from .kdb import KdbHighOrderFeatureEncoder
from sklearn.preprocessing import OneHotEncoder
from pandas import read_csv
import numpy as np

class DataUtils:
    """
    useful data utils for the preparation before training.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.data_size = len(x)
        self.num_features = x.shape[1]

        yunique, ycounts = np.unique(y, return_counts=True)
        self.num_classes = len(yunique)
        self.class_counts = ycounts
        self.feature_uniques = [len(np.unique(x[:,i])) for i in range(self.num_features)]
        
        self.constraint_positions = None
        self._kdbe = None

        self.__kdbe_x = None

    def get_categories(self, idxs=None):
        if idxs != None:
            return [self._kdbe.ohe_.categories_[i] for i in idxs]
        return self._kdbe.ohe_.categories_

    def get_kdbe_x(self, k=0, dense_format=True) -> np.ndarray:
        if self.__kdbe_x is not None:
            return self.__kdbe_x
        if self._kdbe == None:
            self._kdbe = KdbHighOrderFeatureEncoder()
            self._kdbe.fit(self.x, self.y, k=k)
        kdbex = self._kdbe.transform(self.x)
        if dense_format:
            kdbex = kdbex.todense()
        self.__kdbe_x = kdbex
        self.constraint_positions = self._kdbe.constraints_
        return kdbex
    
    def clear(self):
        self._kdbe = None
        self.__kdbe_x = None
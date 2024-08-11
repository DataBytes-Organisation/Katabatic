import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from pandas import read_csv
from .kdb import KdbHighOrderFeatureEncoder

class SoftmaxWeight(tf.keras.constraints.Constraint):
    def __init__(self, feature_uniques):
        idxs = math_ops.cumsum([0] + feature_uniques)
        self.feature_idxs = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
  
    def __call__(self, w):     
        w_new = [
            math_ops.log(tf.nn.softmax(w[i:j,:], axis=0))
            for i, j in self.feature_idxs
        ]
        return tf.concat(w_new, 0)
  
    def get_config(self):
        return {'feature_idxs': self.feature_idxs}

def elr_loss(KL_LOSS):
    def loss(y_true, y_pred):
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) + KL_LOSS
    return loss

def get_lr(input_dim, output_dim, constraint=None, KL_LOSS=0):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(output_dim, input_dim=input_dim, activation='softmax', kernel_constraint=constraint)
    ])
    model.compile(loss=elr_loss(KL_LOSS), optimizer='adam', metrics=['accuracy'])
    return model

def sample(*arrays, n=None, frac=None, random_state=None):
    if n is None and frac is None:
        raise ValueError('You must specify one of frac or n.')
    
    random = np.random.default_rng(random_state)
    arr0 = arrays[0]
    original_size = len(arr0)
    
    if n is None:
        n = int(original_size * frac)

    idxs = random.choice(original_size, n, replace=False)
    
    if len(arrays) > 1:
        return tuple(arr[idxs] for arr in arrays)
    else:
        return arr0[idxs]

DEMO_DATASETS = {
    'adult': {
        'link': 'https://raw.githubusercontent.com/chriszhangpodo/discretizedata/main/adult-dm.csv',
        'params': {'dtype': int}
    },
    'adult-raw': {
        'link': 'https://drive.google.com/uc?export=download&id=1iA-_qIC1xKQJ4nL2ugX1_XJQf8__xOY0',
        'params': {}
    }
}

def get_demo_data(name='adult'):
    if name not in DEMO_DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    dataset = DEMO_DATASETS[name]
    return read_csv(dataset['link'], **dataset['params'])

class DataUtils:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.data_size = len(x)
        self.num_features = x.shape[1]

        yunique, ycounts = np.unique(y, return_counts=True)
        self.num_classes = len(yunique)
        self.class_counts = ycounts
        self.feature_uniques = [len(np.unique(x[:, i])) for i in range(self.num_features)]
        
        self.constraint_positions = None
        self._kdbe = None
        self.__kdbe_x = None

    def get_categories(self, idxs=None):
        if self._kdbe is None:
            raise ValueError("KdbHighOrderFeatureEncoder not initialized. Call get_kdbe_x first.")
        if idxs is not None:
            return [self._kdbe.ohe_.categories_[i] for i in idxs]
        return self._kdbe.ohe_.categories_

    def get_kdbe_x(self, k=0, dense_format=True):
        if self.__kdbe_x is not None:
            return self.__kdbe_x
        if self._kdbe is None:
            self._kdbe = KdbHighOrderFeatureEncoder()
            self._kdbe.fit(self.x, self.y, k=k)
        kdbex = self._kdbe.transform(self.x)
        if dense_format and hasattr(kdbex, 'todense'):
            kdbex = kdbex.todense()
        self.__kdbe_x = kdbex
        self.constraint_positions = self._kdbe.constraints_
        return kdbex
    
    def clear(self):
        self._kdbe = None
        self.__kdbe_x = None

from ..kdb import *
from ..utils import *
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from scipy.special import softmax
from .ganblr import GANBLR

def get_weight(group_loss):
    weight = softmax(1- softmax(group_loss))
    #weight = (1 - softmax(group_loss))/len(group_loss)
    #weight = softmax(1000 - group_loss)
    return weight

class GANBLR_MUG_UNIT(GANBLR):
    def init_unit(self, x, y, k, batch_size=32):
        x = self._ordinal_encoder.fit_transform(x)
        y = self._label_encoder.fit_transform(y).astype(int)
        d = DataUtils(x, y)
        self._d = d
        self.k = k
        self.batch_size = batch_size

    def run_one_epoch(self, x, y, syn_x):
        x = self._ordinal_encoder.transform(x)
        y = self._label_encoder.transform(y).astype(int)
        d = DataUtils(x, y)
        self._d = d
        batch_size = self.batch_size

        discriminator_label = np.hstack([np.ones(d.data_size), np.zeros(len(syn_x))])
        discriminator_input = np.vstack([x, syn_x])
        disc_input, disc_label = sample(discriminator_input, discriminator_label, frac=0.8)
        disc = self._discrim()
        d_history = disc.fit(disc_input, disc_label, batch_size=batch_size, epochs=1, verbose=0).history
        prob_fake = disc.predict(x, verbose=0)
        sim_loss = distance.euclidean(x.ravel(), syn_x.ravel())*0.1
        ls = np.mean(-np.log(np.subtract(1, prob_fake))) + sim_loss
        g_history = self._run_generator(loss=ls).history
        #syn_data = self._sample(verbose=0)
        return self, sim_loss, d_history, g_history


class GANBLR_MUG:
    """
    The GANBLR_MUG Model.
    """
    def __init__(self) -> None:
        self._d = None
        #self.__gen_weights = None
        self.batch_size = None
        self.epochs = None
        self.k = None
        #self.constraints = None
        self._units = None
        self._candidate_labels = None
        self._column_names = None
        self._weight = None
        self._ordinal_encoder = None

    def _init_units(self, data_frame, candidate_labels=None):
        '''
        Parameters
        ----------
        data_frame : array_like of shape (n_samples, n_features)
            Dataset to fit the model. The data should be discrete.

        candidate_labels : array_like of shape (n_labels,), default=None
            Index of candidate labels of the dataset. If `None`, all the features will be used as candidate labels.
        '''
        if candidate_labels is None:
            num_units = data_frame.shape[1]
            candidate_labels = np.arange(num_units).astype(int)
        else:
            candidate_label_idxs = []
            for label in candidate_labels:
                if isinstance(label, str):
                    label_idx = data_frame.columns.get_loc(label)
                    candidate_label_idxs.append(label_idx)
                elif isinstance(label, int):
                    candidate_label_idxs.append(label)
                else:
                    raise Exception(f"Invalid Value in Arugument `candidate_labels`, `{label}` is not a valid column name or index.")

            num_units = len(candidate_label_idxs)
            candidate_labels = np.array(candidate_label_idxs).astype(int)

        units = {idx:GANBLR_MUG_UNIT() for idx in candidate_labels}

        self._ordinal_encoder = OrdinalEncoder().fit(data_frame)
        self._candidate_labels = candidate_labels
        self._units = units

        return units


    def fit(self, data_frame, candidate_labels=None, k=0, batch_size=32, epochs=10, warmup_epochs=2, verbose=1):
        '''
        Fit the model to the given data.

        Parameters
        ----------
        data_frame : array_like of shape (n_samples, n_features)
            Dataset to fit the model. The data should be discrete.

        candidate_labels : array_like of shape (n_labels,), default=None
            Index of candidate labels of the dataset. If `None`, all the features will be used as candidate labels.

        k : int, default=0
            Parameter k of ganblr model. Must be greater than 0. No more than 2 is Suggested.

        batch_size : int, default=32
            Size of the batch to feed the model at each step.

        epochs : int, default=0
            Number of epochs to use during training.

        warmup_epochs : int, default=1
            Number of epochs to use in warmup phase. Defaults to :attr:`1`.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        Returns
        -------
        self : object
            Fitted model.
        '''
        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        self.k = k
        self.batch_size = batch_size
        units = self._init_units(data_frame, candidate_labels)
        weight = self._warmup_run(data_frame, k, batch_size, warmup_epochs, verbose=verbose)
        syn_data = self._weighted_sample(weight, verbose=0)

        for i in range(epochs):
            group_loss = []
            for label_idx, unit in units.items():
                X, y = self._split_dataset(data_frame.values, label_idx)
                syn_X, syn_y = self._split_dataset(syn_data, label_idx)
                unit, sim_loss, d_history, g_history = unit.run_one_epoch(X, y, syn_X)
                group_loss.append(sim_loss)

                print(f"E{i}U{label_idx}: G_loss = {g_history['loss'][0]:.6f}, D_loss = {d_history['loss'][0]:.6f}, sim_loss={sim_loss:.6f}")
            weight = get_weight(group_loss)

            self._weight = weight
            print(f'E{i} weight: {np.round(weight, 2).tolist()}')
            syn_data = self._weighted_sample(weight, verbose=0)

        return self

    def evaluate(self, test_data, label_idx=None, model='lr') -> float:
        """
        Perform a TSTR(Training on Synthetic data, Testing on Real data) evaluation.

        Parameters
        ----------
        test_data : array_like
            Test dataset.

        model : str or object
            The model used for evaluate. Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.

        Return:
        --------
        accuracy_score : float.

        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score

        eval_model = None
        models = dict(
            lr=LogisticRegression,
            rf=RandomForestClassifier,
            mlp=MLPClassifier
        )
        if model in models.keys():
            eval_model = models[model]()
        elif hasattr(model, 'fit') and hasattr(model, 'predict'):
            eval_model = model
        else:
            raise Exception("Invalid Arugument `model`, Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.")

        oe = self._units[label_idx]._ordinal_encoder
        le = self._units[label_idx]._label_encoder
        d  = self._units[label_idx]._d

        synthetic_data = self._weighted_sample(self._weight, verbose=0)
        synthetic_x, synthetic_y = self._split_dataset(synthetic_data, label_idx)
        x, y = self._split_dataset(test_data.values, label_idx)
        x_test = oe.transform(x)
        y_test = le.transform(y)

        categories = d.get_categories()
        pipline = Pipeline([
    ('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore', sparse_output=False)), 
    ('model',  eval_model)
        ])
        pipline.fit(synthetic_x, synthetic_y)
        pred = pipline.predict(x_test)
        return accuracy_score(y_test, pred)

    def sample(self, size=None, verbose=1) -> np.ndarray:
        """
        Generate synthetic data.

        Parameters
        ----------
        size : int or None
            Size of the data to be generated. set to `None` to make the size equal to the size of the training set.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        Return:
        -----------------
        synthetic_samples : np.ndarray
            Generated synthetic data.
        """
        ordinal_data = self._weighted_sample(self._weight, size, verbose)
        ordinal_data = self._ordinal_encoder.inverse_transform(ordinal_data)
        return ordinal_data

    def _weighted_sample(self, weight, size=None, verbose=1) -> np.ndarray:
        samples = []
        sample_size = self._units[0]._d.data_size if size is None else size

        sizes = np.round(weight * sample_size).astype(int)
        if sizes.sum() != sample_size:
            sizes[np.argmax(sizes)] += sample_size - sizes.sum()

        for _size, (idx, unit) in zip(sizes, self._units.items()):
            result = unit._sample(_size, verbose)

            sorted_result = self._reindex_dataset(result, idx)
            samples.append(sorted_result)
            #print(f"sizes: {sorted_result.shape}")
        return np.vstack(samples)

    def _split_dataset(self, data, label_idx):
        '''
        assume idx = 2
        this method convert [x0 x1 x2 x3 y] to [x0 x1 x3 y], x2
        '''
        feature_idxs = np.delete(np.arange(data.shape[1]), label_idx).astype(int)
        X = data[:,feature_idxs]
        y = data[:,label_idx]
        return X, y

    def _reindex_dataset(self, data, label_idx):
        '''
        assume idx = 2
        this method convert [x0 x1 x3 y x2] to [x0 x1 x2 x3 y]
        '''
        feature_idxs = list(range(data.shape[1]-1))
        feature_idxs.insert(label_idx, data.shape[1] - 1)

        return data[:,feature_idxs]

    def _warmup_run(self, data_frame, k, batch_size, epochs, verbose=None):
        if verbose:
            print(f"warmup run:")

        group_loss = []
        for label_idx, unit in self._units.items():
            X, y = self._split_dataset(data_frame.values, label_idx)

            unit.init_unit(X, y, k, batch_size)
            unit._warmup_run(epochs, verbose=verbose)
            syn_data = unit._sample(verbose=0)
            syn_X, syn_y = self._split_dataset(syn_data, label_idx)
            unit, sim_loss, d_history, g_history = unit.run_one_epoch(X, y, syn_X)
            group_loss.append(sim_loss)

        #print(group_loss)
        weight = get_weight(group_loss)
        if verbose:
            print(f'weight after warmup: {np.round(weight, 2).tolist()}')
        return weight

    def _run_generator(self, loss):
        d = self._d
        ohex = d.get_kdbe_x(self.k)
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(d.num_classes, input_dim=ohex.shape[1], activation='softmax',kernel_constraint=self.constraints))
        model.compile(loss=elr_loss(loss), optimizer='adam', metrics=['accuracy'])
        model.set_weights(self.__gen_weights)
        history = model.fit(ohex, d.y, batch_size=self.batch_size,epochs=1, verbose=0)
        self.__gen_weights = model.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _discrim(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(1, input_dim=self._d.num_features, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

import pgmpy
from .kdb import *
from .kdb import _add_uniform
from .utils import *
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

class GANBLR:
    """
    The GANBLR Model.
    """
    def __init__(self) -> None:
        self._d = None
        self.__gen_weights = None
        self.batch_size = None
        self.epochs = None
        self.k = None
        self.constraints = None
        self._ordinal_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        self._label_encoder   = LabelEncoder()
    
    def fit(self, x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=0):
        '''
        Fit the model to the given data.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features)
            Dataset to fit the model. The data should be discrete.
        
        y : array_like of shape (n_samples,)
            Label of the dataset.

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
        x = self._ordinal_encoder.fit_transform(x)
        y = self._label_encoder.fit_transform(y).astype(int)
        d = DataUtils(x, y)
        self._d = d
        self.k = k
        self.batch_size = batch_size
        if verbose:
            print(f"warmup run:")
        history = self._warmup_run(warmup_epochs, verbose=verbose)
        syn_data = self._sample(verbose=0)
        discriminator_label = np.hstack([np.ones(d.data_size), np.zeros(d.data_size)])
        for i in range(epochs):
            discriminator_input = np.vstack([x, syn_data[:,:-1]])
            disc_input, disc_label = sample(discriminator_input, discriminator_label, frac=0.8)
            disc = self._discrim()
            d_history = disc.fit(disc_input, disc_label, batch_size=batch_size, epochs=1, verbose=0).history
            prob_fake = disc.predict(x, verbose=0)
            ls = np.mean(-np.log(np.subtract(1, prob_fake)))
            g_history = self._run_generator(loss=ls).history
            syn_data = self._sample(verbose=0)
            
            if verbose:
                print(f"Epoch {i+1}/{epochs}: G_loss = {g_history['loss'][0]:.6f}, G_accuracy = {g_history['accuracy'][0]:.6f}, D_loss = {d_history['loss'][0]:.6f}, D_accuracy = {d_history['accuracy'][0]:.6f}")
        return self
        
    def evaluate(self, x, y, model='lr') -> float:
        """
        Perform a TSTR(Training on Synthetic data, Testing on Real data) evaluation.

        Parameters
        ----------
        x, y : array_like
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

        synthetic_data = self._sample()
        synthetic_x, synthetic_y = synthetic_data[:,:-1], synthetic_data[:,-1]
        x_test = self._ordinal_encoder.transform(x)
        y_test = self._label_encoder.transform(y)

        categories = self._d.get_categories()
        pipline = Pipeline([('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore')), ('model',  eval_model)]) 
        pipline.fit(synthetic_x, synthetic_y)
        pred = pipline.predict(x_test)
        return accuracy_score(y_test, pred)
    
    def sample(self, size=None, verbose=0) -> np.ndarray:
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
        ordinal_data = self._sample(size, verbose)
        origin_x = self._ordinal_encoder.inverse_transform(ordinal_data[:,:-1])
        origin_y = self._label_encoder.inverse_transform(ordinal_data[:,-1]).reshape(-1,1)
        return np.hstack([origin_x, origin_y])
        
    def _sample(self, size=None, verbose=0) -> np.ndarray:
        """
        Generate synthetic data in ordinal encoding format
        """
        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        #basic varibles
        d = self._d
        feature_cards = np.array(d.feature_uniques)
        #ensure sum of each constraint group equals to 1, then re concat the probs
        _idxs = np.cumsum([0] + d._kdbe.constraints_.tolist())
        constraint_idxs = [(_idxs[i],_idxs[i+1]) for i in range(len(_idxs)-1)]
        
        probs = np.exp(self.__gen_weights[0])
        cpd_probs = [probs[start:end,:] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p/p.sum(axis=0) for p in cpd_probs])
    
        #assign the probs to the full cpd tables
        idxs = np.cumsum([0] + d._kdbe.high_order_feature_uniques_)
        feature_idxs = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
        have_value_idxs = d._kdbe.have_value_idxs_
        full_cpd_probs = [] 
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            #(n_high_order_feature_uniques, n_classes)
            cpd_prob_ = cpd_probs[start:end,:]
            #(n_all_combination) Note: the order is (*parent, variable)
            have_value_ravel = have_value.ravel()
            #(n_classes * n_all_combination)
            have_value_ravel_repeat = np.hstack([have_value_ravel] * d.num_classes)
            #(n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
            full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
            full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
            #(n_classes * n_parent_combinations, n_variable_unique)
            full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
            full_cpd_prob = _add_uniform(full_cpd_prob, noise=0)
            full_cpd_probs.append(full_cpd_prob)
    
        #prepare node and edge names
        node_names = [str(i) for i in range(d.num_features + 1)]
        edge_names = [(str(i), str(j)) for i,j in d._kdbe.edges_]
        y_name = node_names[-1]
    
        #create TabularCPD objects
        evidences = d._kdbe.dependencies_
        feature_cpds = [
            TabularCPD(str(name), feature_cards[name], table, 
                       evidence=[y_name, *[str(e) for e in evidences]], 
                       evidence_card=[d.num_classes, *feature_cards[evidences].tolist()])
            for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
        ]
        y_probs = (d.class_counts/d.data_size).reshape(-1,1)
        y_cpd = TabularCPD(y_name, d.num_classes, y_probs)
    
        #create kDB model, then sample the data
        model = BayesianNetwork(edge_names)
        model.add_cpds(y_cpd, *feature_cpds)
        sample_size = d.data_size if size is None else size
        result = BayesianModelSampling(model).forward_sample(size=sample_size, show_progress = verbose > 0)
        sorted_result = result[node_names].values
        
        return sorted_result
    
    def _warmup_run(self, epochs, verbose=None):
        d = self._d
        tf.keras.backend.clear_session()
        ohex = d.get_kdbe_x(self.k)
        self.constraints = softmax_weight(d.constraint_positions)
        elr = get_lr(ohex.shape[1], d.num_classes, self.constraints)
        history = elr.fit(ohex, d.y, batch_size=self.batch_size, epochs=epochs, verbose=verbose)
        self.__gen_weights = elr.get_weights()
        tf.keras.backend.clear_session()
        return history

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
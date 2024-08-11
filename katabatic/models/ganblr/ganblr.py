import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from .utils import DataUtils, get_lr, sample, elr_loss
from .kdb import KdbHighOrderFeatureEncoder

class GANBLR:
    def __init__(self):
        self._d = None
        self.__gen_weights = None
        self.batch_size = 32
        self.epochs = 10
        self.k = 0
        self.constraints = None
        self._ordinal_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        self._label_encoder = LabelEncoder()
    
    def fit(self, x, y, k=0, batch_size=32, epochs=10, warmup_epochs=1, verbose=0):
        x = self._ordinal_encoder.fit_transform(x)
        y = self._label_encoder.fit_transform(y).astype(int)
        self._d = DataUtils(x, y)
        self.k = k
        self.batch_size = batch_size
        self.epochs = epochs

        if verbose:
            print("Warmup run:")
        self._warmup_run(warmup_epochs, verbose=verbose)

        discriminator = self._create_discriminator()
        
        for epoch in range(epochs):
            syn_data = self._sample(verbose=0)
            discriminator_input = np.vstack([x, syn_data[:,:-1]])
            discriminator_label = np.hstack([np.ones(self._d.data_size), np.zeros(self._d.data_size)])
            disc_input, disc_label = sample(discriminator_input, discriminator_label, frac=0.8)
            
            d_history = discriminator.fit(disc_input, disc_label, batch_size=batch_size, epochs=1, verbose=0)
            prob_fake = discriminator.predict(x, verbose=0)
            ls = np.mean(-np.log(np.subtract(1, prob_fake)))
            g_history = self._run_generator(loss=ls)
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}: G_loss = {g_history['loss'][0]:.6f}, G_accuracy = {g_history['accuracy'][0]:.6f}, D_loss = {d_history.history['loss'][0]:.6f}, D_accuracy = {d_history.history['accuracy'][0]:.6f}")
        
        return self

    def _warmup_run(self, epochs, verbose=0):
        d = self._d
        tf.keras.backend.clear_session()
        ohex = d.get_kdbe_x(self.k)
        self.constraints = self._create_softmax_weight(d.constraint_positions)
        elr = get_lr(ohex.shape[1], d.num_classes, self.constraints)
        history = elr.fit(ohex, d.y, batch_size=self.batch_size, epochs=epochs, verbose=verbose)
        self.__gen_weights = elr.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _run_generator(self, loss):
        d = self._d
        ohex = d.get_kdbe_x(self.k)
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(d.num_classes, input_dim=ohex.shape[1], activation='softmax', kernel_constraint=self.constraints)
        ])
        model.compile(loss=elr_loss(loss), optimizer='adam', metrics=['accuracy'])
        model.set_weights(self.__gen_weights)
        history = model.fit(ohex, d.y, batch_size=self.batch_size, epochs=1, verbose=0)
        self.__gen_weights = model.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _create_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_dim=self._d.num_features, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def _create_softmax_weight(feature_uniques):
        idxs = np.cumsum([0] + feature_uniques.tolist())
        feature_idxs = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
        
        class SoftmaxWeight(tf.keras.constraints.Constraint):
            def __call__(self, w):
                w_new = [tf.math.log(tf.nn.softmax(w[i:j,:], axis=0)) for i, j in feature_idxs]
                return tf.concat(w_new, 0)
        
        return SoftmaxWeight()

    def _sample(self, size=None, verbose=0):
        d = self._d
        feature_cards = np.array(d.feature_uniques)
        _idxs = np.cumsum([0] + d._kdbe.constraints_.tolist())
        constraint_idxs = [(_idxs[i], _idxs[i+1]) for i in range(len(_idxs)-1)]
        
        probs = np.exp(self.__gen_weights[0])
        cpd_probs = [probs[start:end,:] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p/p.sum(axis=0) for p in cpd_probs])
    
        idxs = np.cumsum([0] + d._kdbe.high_order_feature_uniques_)
        feature_idxs = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
        have_value_idxs = d._kdbe.have_value_idxs_
        full_cpd_probs = [] 
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            cpd_prob_ = cpd_probs[start:end,:]
            have_value_ravel = have_value.ravel()
            have_value_ravel_repeat = np.hstack([have_value_ravel] * d.num_classes)
            full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
            full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
            full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
            full_cpd_prob = self._add_uniform(full_cpd_prob, noise=0)
            full_cpd_probs.append(full_cpd_prob)
    
        node_names = [str(i) for i in range(d.num_features + 1)]
        edge_names = [(str(i), str(j)) for i,j in d._kdbe.edges_]
        y_name = node_names[-1]
    
        evidences = d._kdbe.dependencies_
        feature_cpds = [
            TabularCPD(str(name), feature_cards[name], table, 
                       evidence=[y_name, *[str(e) for e in evidences]], 
                       evidence_card=[d.num_classes, *feature_cards[evidences].tolist()])
            for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
        ]
        y_probs = (d.class_counts/d.data_size).reshape(-1,1)
        y_cpd = TabularCPD(y_name, d.num_classes, y_probs)
    
        model = BayesianNetwork(edge_names)
        model.add_cpds(y_cpd, *feature_cpds)
        sample_size = d.data_size if size is None else size
        result = BayesianModelSampling(model).forward_sample(size=sample_size, show_progress=verbose > 0)
        sorted_result = result[node_names].values
        
        return sorted_result

    @staticmethod
    def _add_uniform(array, noise=1e-5):
        sum_by_col = np.sum(array, axis=0)
        zero_idxs = (array == 0).astype(int)
        nunique = array.shape[0]
        result = np.zeros_like(array, dtype='float')
        for i in range(array.shape[1]):
            if sum_by_col[i] == 0:
                result[:,i] = array[:,i] + 1./nunique
            elif noise != 0:
                result[:,i] = array[:,i] + noise * zero_idxs[:,i]
            else:
                result[:,i] = array[:,i]
        return result

    def evaluate(self, x, y, model='lr'):
        models = {
            'lr': LogisticRegression,
            'rf': RandomForestClassifier,
            'mlp': MLPClassifier
        }
        
        if model in models:
            eval_model = models[model]()
        elif hasattr(model, 'fit') and hasattr(model, 'predict'):
            eval_model = model
        else:
            raise ValueError("Invalid model argument")

        synthetic_data = self._sample()
        synthetic_x, synthetic_y = synthetic_data[:,:-1], synthetic_data[:,-1]
        x_test = self._ordinal_encoder.transform(x)
        y_test = self._label_encoder.transform(y)

        categories = self._d.get_categories()
        pipeline = Pipeline([
            ('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore')),
            ('model',  eval_model)
        ]) 
        pipeline.fit(synthetic_x, synthetic_y)
        pred = pipeline.predict(x_test)
        return accuracy_score(y_test, pred)
    
    def sample(self, size=None, verbose=0):
        ordinal_data = self._sample(size, verbose)
        origin_x = self._ordinal_encoder.inverse_transform(ordinal_data[:,:-1])
        origin_y = self._label_encoder.inverse_transform(ordinal_data[:,-1]).reshape(-1,1)
        return np.hstack([origin_x, origin_y])

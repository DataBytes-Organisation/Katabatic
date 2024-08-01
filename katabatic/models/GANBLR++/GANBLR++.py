import numpy as np
import pandas as pd
from ganblr.utils import get_demo_data
from ganblr import GANBLRPP
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

class DataHandler:
    def __init__(self, dataset_name='adult-raw'):
        self.df = get_demo_data(dataset_name)
        self.X_train, self.X_test, self.y_train, self.y_test = self._split_data()
        self.numerical_columns = self._identify_numerical_columns()

    def _split_data(self):
        x, y = self.df.values[:, :-1], self.df.values[:, -1]
        return train_test_split(x, y, test_size=0.5)

    def _identify_numerical_columns(self):
        def is_numerical(dtype):
            return dtype.kind in 'iuf'
        column_is_numerical = self.df.dtypes.apply(is_numerical).values
        return np.argwhere(column_is_numerical).ravel()

class GANBLRPPModel:
    def __init__(self, numerical_columns):
        self.model = GANBLRPP(numerical_columns)

    def train(self, X_train, y_train, epochs=10):
        self.model.fit(X_train, y_train, epochs=epochs)

    def generate_data(self, size=1000):
        return self.model.sample(size)

    def evaluate(self, X_test, y_test, model_type='lr'):
        return self.model.evaluate(X_test, y_test, model=model_type)

class Evaluator:
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.scaler = StandardScaler()
        self.lbe = LabelEncoder()
        self._prepare_data()

    def _prepare_data(self):
        catgorical_columns = list(set(range(self.data_handler.X_train.shape[1])) - set(self.data_handler.numerical_columns))

        X_train_ohe = self.ohe.fit_transform(self.data_handler.X_train[:, catgorical_columns])
        X_test_ohe = self.ohe.transform(self.data_handler.X_test[:, catgorical_columns])
        X_train_num = self.data_handler.X_train[:, self.data_handler.numerical_columns]
        X_test_num = self.data_handler.X_test[:, self.data_handler.numerical_columns]

        self.X_train_concat = self.scaler.fit_transform(np.hstack([X_train_num, X_train_ohe]))
        self.X_test_concat = self.scaler.transform(np.hstack([X_test_num, X_test_ohe]))

        self.y_train_lbe = self.lbe.fit_transform(self.data_handler.y_train)
        self.y_test_lbe = self.lbe.transform(self.data_handler.y_test)

    def evaluate_trtr(self):
        models = {
            'LR': LogisticRegression(),
            'RF': RandomForestClassifier(),
            'MLP': MLPClassifier()
        }

        results = {}
        for name, model in models.items():
            model.fit(self.X_train_concat, self.data_handler.y_train)
            score = model.score(self.X_test_concat, self.data_handler.y_test)
            results[name] = score

        return results

    def compare_results(self, tstr_scores, trtr_scores):
        df_evaluate = pd.DataFrame([
            ['TSTR', tstr_scores['LR'], tstr_scores['RF'], tstr_scores['MLP']],
            ['TRTR', trtr_scores['LR'], trtr_scores['RF'], trtr_scores['MLP']]
        ], columns=['Evaluated Item', 'LR', 'RF', 'MLP'])

        print(df_evaluate)

# Usage
data_handler = DataHandler()
ganblr_model = GANBLRPPModel(data_handler.numerical_columns)

# Train the GANBLR model
ganblr_model.train(data_handler.X_train, data_handler.y_train, epochs=10)

# Generate synthetic data
synthetic_data = ganblr_model.generate_data()
print("Synthetic Data Sample:\n", pd.DataFrame(synthetic_data, columns=data_handler.df.columns).head(10))

# Evaluate using TSTR
tstr_scores = {
    'LR': ganblr_model.evaluate(data_handler.X_test, data_handler.y_test, model_type='lr'),
    'RF': ganblr_model.evaluate(data_handler.X_test, data_handler.y_test, model_type='rf'),
    'MLP': ganblr_model.evaluate(data_handler.X_test, data_handler.y_test, model_type='mlp')
}

# Evaluate using TRTR
evaluator = Evaluator(data_handler)
trtr_scores = evaluator.evaluate_trtr()

# Compare Results
evaluator.compare_results(tstr_scores, trtr_scores)





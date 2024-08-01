from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Input must be dataframes
def evaluate(X_synthetic, y_synthetic, X_real, y_real):
    # TODO: error handling, data validation

    # Convert from pandas objects into numpy arrays.
    X_synthetic = X_synthetic.to_numpy()
    y_synthetic = y_synthetic.to_numpy().ravel()
    X_real = X_real.to_numpy()
    y_real = y_real.to_numpy().ravel()
    
    y_train = y_synthetic.astype('int')
    # TSTR Evaluation using Log Reg
    model = LogisticRegression(max_iter=200)
    model.fit(X_synthetic, y_train)
    y_pred = model.predict(X_real)

    return accuracy_score(y_real, y_pred)
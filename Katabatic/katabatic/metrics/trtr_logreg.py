from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate(X_synthetic, y_synthetic, X_real, y_real):

    # TODO: error handling, data validation
    X_synthetic = X_synthetic.to_numpy()
    y_synthetic = y_synthetic.to_numpy().ravel()
    X_real = X_real.to_numpy()
    y_real = y_real.to_numpy().ravel()

    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.33, random_state=42)
    
    le = LabelEncoder() # Use labelencoder to convert strings to values
    le.fit(np.unique(y_train))   
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # TSTR Evaluation using Log Reg
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred)
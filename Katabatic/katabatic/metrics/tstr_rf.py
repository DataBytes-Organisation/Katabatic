from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

def evaluate(X_synthetic, y_synthetic, X_real, y_real):

    # TODO: error handling, data validation
    X_synthetic = X_synthetic.to_numpy()
    y_synthetic = y_synthetic.to_numpy().ravel()
    X_real = X_real.to_numpy()
    y_real = y_real.to_numpy().ravel()
    
    le = LabelEncoder() # Use labelencoder to convert strings to values
    le.fit(np.unique(y_synthetic))   # TODO: Combine both y_synth and y_real values here
    y_synthetic = le.transform(y_synthetic)
    y_real = le.transform(y_real)

    # TSTR Evaluation using Random Forest
    model = RandomForestClassifier()    # LogisticRegression(max_iter=200)
    model.fit(X_synthetic, y_synthetic)
    y_pred = model.predict(X_real)

    return accuracy_score(y_real, y_pred)
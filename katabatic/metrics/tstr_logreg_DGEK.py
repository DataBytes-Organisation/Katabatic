import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluate(X_synthetic, y_synthetic, X_real, y_real):
    # Ensure inputs are dataframes
    if not all(isinstance(df, pd.DataFrame) for df in [X_synthetic, X_real]):
        raise ValueError("X_synthetic and X_real must be dataframes.")
    if not all(isinstance(s, pd.Series) for s in [y_synthetic, y_real]):
        raise ValueError("y_synthetic and y_real must be series.")

    # Convert from pandas objects into numpy arrays
    X_synthetic = X_synthetic.to_numpy()
    y_synthetic = y_synthetic.to_numpy().ravel()
    X_real = X_real.to_numpy()
    y_real = y_real.to_numpy().ravel()
    
    # Ensure y_synthetic and y_real are integers
    y_train = y_synthetic.astype('int')
    y_test = y_real.astype('int')

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_synthetic, y_train)
    y_pred = model.predict(X_real)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Return accuracy score rounded to 4 decimal places
    return round(accuracy, 4)

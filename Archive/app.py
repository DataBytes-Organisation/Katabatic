from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('error.html', error="No file uploaded")

        # Read the uploaded CSV file
        file = request.files['file']
        df = pd.read_csv(file)

        # Log the dataset for debugging
        print("Initial Dataset Head:")
        print(df.head())

        # Validate dataset size
        if df.shape[1] < 2:
            return render_template('error.html', error="Dataset must have at least two columns (features and target)")

        # Convert non-numeric values in features to NaN and drop them
        X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce')

        # Ensure the target column is handled correctly
        y = df.iloc[:, -1]
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        else:
            y = pd.to_numeric(y, errors='coerce')

        # Combine cleaned features and target for validation
        df_cleaned = pd.concat([X, pd.Series(y, name='target')], axis=1).dropna()

        # Log cleaned dataset
        print("Cleaned Dataset Head:")
        print(df_cleaned.head())

        # Ensure the cleaned dataset is still valid
        if df_cleaned.empty or df_cleaned.shape[1] < 2:
            return render_template('error.html', error="Dataset is empty or invalid after cleaning. Ensure your dataset is properly formatted.")

        # Separate features (X) and target (y)
        X = df_cleaned.iloc[:, :-1].values
        y = df_cleaned.iloc[:, -1].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build a simple neural network
        model = Sequential([
            Dense(16, input_dim=X_train.shape[1], activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Set default epochs
        epochs = int(request.form.get('epochs', 10))

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=8, verbose=1)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        return render_template('success.html', accuracy=f"{accuracy:.2%}")

    except Exception as e:
        print(f"Error during processing: {e}")
        return render_template('error.html', error=f"Data preprocessing error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)

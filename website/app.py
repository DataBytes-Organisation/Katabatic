from flask import Flask, request, jsonify, render_template, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from ganblrplus import GANBLRPlusPlus
import os
import io

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
UPLOAD_FOLDER = 'uploads'
SYNTHETIC_DATA_FOLDER = 'synthetic_data'  # Folder to store synthetic data CSVs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SYNTHETIC_DATA_FOLDER, exist_ok=True)

app = Flask(__name__)




def preprocess_data(file):
    """Preprocess the uploaded CSV file."""
    try:
        # Load the CSV file into a DataFrame
        data = pd.read_csv(file)

        # Drop the 'Id' column if it exists
        if 'Id' in data.columns:
            data = data.drop(columns=['Id'])

        # One-hot encode the 'Species' column if it exists
        if 'Species' in data.columns:
            encoder = OneHotEncoder(sparse=False)
            species_encoded = encoder.fit_transform(data[['Species']])
            encoded_columns = encoder.get_feature_names_out(['Species'])
            species_encoded_df = pd.DataFrame(species_encoded, columns=encoded_columns)
            data = pd.concat([data.drop(columns=['Species']), species_encoded_df], axis=1)

        return data
    except Exception as e:
        raise ValueError(f"Error preprocessing the data: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        return jsonify({'message': 'Dataset uploaded successfully! at ' + filepath, 'filepath': filepath}), 200
    
    return jsonify({'error': 'Invalid file format. Please upload a CSV file.'}), 400


@app.route('/generate', methods=['POST'])
def generate():
    # Get the file path from the request
    filepath = request.json.get('filepath') # Expecting the file path in the request JSON
    epochs = request.json.get('epochs')  
    batch_size = request.json.get('batch_size')  

    if not filepath:
        return jsonify({'error': 'Invalid or missing file path. Please upload a file first.'}), 400

    try:
        # Preprocess the data
        data = preprocess_data(filepath)
        data_array = data.to_numpy()

        # Validate epochs and batch_size
        if not isinstance(epochs, int) or epochs <= 0:
            return jsonify({'error': 'Invalid value for epochs. Must be a positive integer.'}), 400
        if not isinstance(batch_size, int) or batch_size <= 0:
            return jsonify({'error': 'Invalid value for batch_size. Must be a positive integer.'}), 400

        # Train the GANBLR++ model
        model = GANBLRPlusPlus(input_dim=data_array.shape[1], latent_dim=20)
        model.train(data_array, epochs=epochs, batch_size=batch_size)

        # Generate synthetic data
        synthetic_data = model.generate_batch(batch_size)

        # Save synthetic data to a CSV file
        synthetic_data_df = pd.DataFrame(synthetic_data, columns=[f"Feature_{i+1}" for i in range(synthetic_data.shape[1])])
        synthetic_data_filename = f"synthetic_data_{epochs}_{batch_size}.csv"
        synthetic_data_path = os.path.join(SYNTHETIC_DATA_FOLDER, synthetic_data_filename)
        synthetic_data_df.to_csv(synthetic_data_path, index=False)

        return jsonify({
            "message": "Synthetic data generated and saved successfully.",
            "csv_url": url_for('static', filename=f'synthetic_data/{synthetic_data_filename}')
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)
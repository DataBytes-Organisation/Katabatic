from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from io import BytesIO
from katabatic.models.ctgan.ctgan_adapter import CtganAdapter
import time

app = Flask(__name__)
dataset = None
synthetic_data = None
training_status = {"status": "idle", "message": ""}  # Global status variable


@app.route('/')
def index():
    """
    Render the homepage.
    """
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle dataset upload and column extraction.
    """
    global dataset
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['dataset']
    try:
        dataset = pd.read_csv(file)
        columns = dataset.columns.tolist()
        return jsonify({'columns': columns})
    except Exception as e:
        return jsonify({'error': f'Error reading dataset: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """
    Train the model and generate synthetic data.
    """
    global dataset, synthetic_data, training_status
    if dataset is None:
        return jsonify({'error': 'No dataset uploaded. Please upload a dataset first.'}), 400

    try:
        target_columns = request.form.getlist('target_columns[]')
        num_samples = int(request.form.get('num_samples', 0))

        if num_samples <= 0:
            return jsonify({'error': 'Number of synthetic rows must be greater than 0.'}), 400

        # Step 1: Initializing Model
        training_status['status'] = 'in_progress'
        training_status['message'] = 'Initializing the model...'
        time.sleep(1)

        # Step 2: Training Model
        training_status['message'] = 'Training the model...'
        time.sleep(1)
        model = CtganAdapter(epochs=5)
        model.fit(dataset)

        # Step 3: Generating Data
        training_status['message'] = 'Generating synthetic data...'
        time.sleep(1)
        synthetic_data = model.generate(num_samples)

        if synthetic_data is None or synthetic_data.empty:
            training_status['status'] = 'error'
            training_status['message'] = 'Synthetic data generation failed.'
            return jsonify({'error': 'Failed to generate synthetic data.'}), 500

        synthetic_data = pd.DataFrame(synthetic_data, columns=dataset.columns)

        # Filter the data to include only selected columns
        if target_columns:
            synthetic_data = synthetic_data[target_columns]

        # Finalizing Training Status
        training_status['status'] = 'completed'
        training_status['message'] = 'Training and data generation completed successfully.'

        # Convert to JSON for frontend display
        synthetic_data_json = synthetic_data.to_dict(orient='records')
        return jsonify({'synthetic_data': synthetic_data_json})

    except Exception as e:
        training_status['status'] = 'error'
        training_status['message'] = f'Error during training: {str(e)}'
        return jsonify({'error': f'Error during synthetic data generation: {str(e)}'}), 500

@app.route('/training_status', methods=['GET'])
def training_status_endpoint():
    """
    Return the current training status.
    """
    global training_status
    return jsonify(training_status)


@app.route('/download', methods=['GET'])
def download():
    """
    Allow users to download the generated synthetic data.
    """
    global synthetic_data
    if synthetic_data is None:
        return jsonify({'error': 'No synthetic data available to download.'}), 400

    output = BytesIO()
    synthetic_data.to_csv(output, index=False)
    output.seek(0)

    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='synthetic_data.csv')


if __name__ == '__main__':
    app.run(debug=False)

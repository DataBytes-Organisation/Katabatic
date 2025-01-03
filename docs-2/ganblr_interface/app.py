from flask import Flask, render_template, request, redirect, url_for
import subprocess
import os

app = Flask(__name__)

# Define directories for uploads and visualizations
UPLOAD_FOLDER = "uploaded_datasets"
VISUALIZATION_DIR = "static/visualizations"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    """
    Renders the homepage.
    """
    return render_template('index.html', status="", outputs={})

def run_script(script_name, args=None, output_flag=None):
    """
    Helper function to run a script and return its output.
    """
    try:
        command = ["python", script_name]
        if args:
            command.extend(args)
        if output_flag:
            command.extend(["--output", VISUALIZATION_DIR])
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "output": e.stderr}

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    """
    Handles file upload and runs the load_credit_dataset.py script.
    """
    file = request.files.get('datasetFile')
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = run_script("load_credit_dataset.py", args=["--file", file_path])
        return render_template('index.html', status="Dataset loaded successfully!" if result['status'] == "success" else "Failed to load dataset.", outputs={"load_dataset": result})
    else:
        return render_template('index.html', status="No file selected. Please upload a dataset file.", outputs={})

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    """
    Handles preprocessing file upload and runs the preprocess_credit_data.py script.
    """
    file = request.files.get('preprocessFile')
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = run_script("preprocess_credit_data.py", args=["--file", file_path])
        return render_template('index.html', status="Data preprocessing complete!" if result['status'] == "success" else "Failed to preprocess data.", outputs={"preprocess_data": result})
    else:
        return render_template('index.html', status="No file selected. Please upload a preprocess file.", outputs={})

@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Handles file upload for training and runs the train_ganblr_credit.py script.
    """
    x_train_file = request.files.get('X_train_file')
    y_train_file = request.files.get('Y_train_file')

    if x_train_file and y_train_file:
        x_train_path = os.path.join(app.config['UPLOAD_FOLDER'], x_train_file.filename)
        y_train_path = os.path.join(app.config['UPLOAD_FOLDER'], y_train_file.filename)
        
        x_train_file.save(x_train_path)
        y_train_file.save(y_train_path)
        
        result = run_script("train_ganblr_credit.py", args=["--x_train", x_train_path, "--y_train", y_train_path])
        return render_template('index.html', status="Model training complete!" if result['status'] == "success" else "Failed to train model.", outputs={"train_model": result})
    else:
        return render_template('index.html', status="Please upload both X_train and Y_train files.", outputs={})

@app.route('/verify_data', methods=['POST'])
def verify_data():
    """
    Handles file upload for verification and runs the verify_synthetic_data.py script.
    """
    file = request.files.get('verifyFile')
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        result = run_script("verify_synthetic_data.py", args=["--file", file_path])
        return render_template('index.html', status="Data verification complete!" if result['status'] == "success" else "Failed to verify data.", outputs={"verify_data": result})
    else:
        return render_template('index.html', status="No file selected. Please upload a verification file.", outputs={})

@app.route('/generate_visualizations', methods=['POST'])
def generate_visualizations():
    """
    Handles file upload for visualizations and runs the visualization scripts.
    """
    real_file = request.files.get('real_data_file')
    synthetic_file = request.files.get('synthetic_data_file')

    if real_file and synthetic_file:
        real_file_path = os.path.join(app.config['UPLOAD_FOLDER'], real_file.filename)
        synthetic_file_path = os.path.join(app.config['UPLOAD_FOLDER'], synthetic_file.filename)
        
        real_file.save(real_file_path)
        synthetic_file.save(synthetic_file_path)
        
        feature_dist_result = run_script("feature-distribution.py", args=["--real", real_file_path, "--synthetic", synthetic_file_path, "--output", VISUALIZATION_DIR])
        correlation_heatmap_result = run_script("correlation-heat-maps.py", args=["--real", real_file_path, "--synthetic", synthetic_file_path, "--output", VISUALIZATION_DIR])

        if feature_dist_result['status'] == "success" and correlation_heatmap_result['status'] == "success":
            return redirect(url_for('visualizations'))
        else:
            return render_template('index.html', status="Failed to generate visualizations. Check script logs for errors.", outputs={"feature_distribution": feature_dist_result, "correlation_heatmaps": correlation_heatmap_result})
    else:
        return render_template('index.html', status="Please upload both real and synthetic dataset files.", outputs={})

@app.route('/visualizations')
def visualizations():
    """
    Displays the generated visualizations dynamically.
    """
    feature_plots = [f for f in os.listdir(VISUALIZATION_DIR) if f.endswith('_comparison.png')]
    heatmap_plots = [f for f in os.listdir(VISUALIZATION_DIR) if 'heatmap' in f]

    return render_template('visualizations.html',
                           status="Here are the visualizations:",
                           feature_plots=feature_plots,
                           heatmap_plots=heatmap_plots,
                           visualization_dir=VISUALIZATION_DIR)

if __name__ == '__main__':
    app.run(debug=True)

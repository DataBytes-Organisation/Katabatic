from flask import Flask, request, render_template_string, redirect, url_for, send_file, flash
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from io import BytesIO
from katabatic.models.meg_DGEK.utils import get_demo_data
from katabatic.models.meg_DGEK.meg_adapter import MegAdapter

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Global Variables
training_data = None
adapter = None
model_trained = False
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEG Synthetic Data Generator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        #training-message {
            display: none;
        }
        .dots {
            display: inline-block;
            font-size: 1.5em;
            animation: dots 1.5s steps(4, end) infinite;
        }
        @keyframes dots {
            0% {
                content: "";
            }
            25% {
                content: ".";
            }
            50% {
                content: "..";
            }
            75% {
                content: "...";
            }
        }
    </style>
    <script>
        function showTrainingMessage() {
            const messageDiv = document.getElementById('training-message');
            messageDiv.style.display = 'block';
        }
    </script>
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center text-primary mb-4">MEG Synthetic Data Generator</h1>

        {% with messages = get_flashed_messages() %}
        {% if messages %}
        <div class="alert alert-success alert-dismissible fade show" role="alert">
            {{ messages[0] }}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>
        {% endif %}
        {% endwith %}

        <div id="training-message" class="alert alert-info">
            Model training in progress. Please wait<span class="dots">...</span>
        </div>

        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Upload Training Data</h5>
                <form action="/upload" method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" class="form-control mb-3" required>
                    <button type="submit" class="btn btn-primary">Upload Data</button>
                </form>
            </div>
        </div>

        <div class="card mt-4">
            <div class="card-body">
                <h5 class="card-title">Load Demo Data</h5>
                <form action="/load_demo" method="POST">
                    <button type="submit" class="btn btn-primary">Load Demo Data</button>
                </form>
            </div>
        </div>

        {% if training_data %}
        <div class="card mt-4">
            <div class="card-body">
                <h5 class="card-title">Train the Model</h5>
                <form action="/train" method="POST" onsubmit="showTrainingMessage()">
                    <button type="submit" class="btn btn-primary">Train Model</button>
                </form>
            </div>
        </div>
        {% endif %}

        {% if model_trained %}
        <div class="card mt-4">
            <div class="card-body">
                <h5 class="card-title">Generate Synthetic Data</h5>
                <form action="/generate" method="POST">
                    <label for="sample_size" class="form-label">Sample Size</label>
                    <input type="number" name="sample_size" id="sample_size" class="form-control mb-3" min="1" required>
                    <button type="submit" class="btn btn-primary">Generate Data</button>
                </form>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    global training_data, model_trained
    return render_template_string(
        HTML_TEMPLATE,
        training_data=training_data is not None,
        model_trained=model_trained
    )

@app.route("/upload", methods=["POST"])
def upload():
    global training_data
    try:
        file = request.files["file"]
        if not file:
            raise ValueError("No file uploaded.")
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        training_data = pd.read_csv(file_path)
        flash("Training data uploaded successfully!")
    except Exception as e:
        flash(f"Error uploading file: {e}")
    return redirect(url_for("index"))

@app.route("/load_demo", methods=["POST"])
def load_demo():
    global training_data
    try:
        training_data = get_demo_data('adult-raw')
        flash("Demo data loaded successfully!")
    except Exception as e:
        flash(f"Error loading demo data: {e}")
    return redirect(url_for("index"))

@app.route("/train", methods=["POST"])
def train():
    global training_data, adapter, model_trained
    try:
        if training_data is None:
            raise ValueError("No training data available.")
        
        X, y = training_data.values[:, :-1], training_data.values[:, -1]
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=42)
        
        adapter = MegAdapter()
        adapter.load_model()
        adapter.fit(X_train, y_train, epochs=5)
        
        model_trained = True
        flash("Model trained successfully!")
    except Exception as e:
        flash(f"Error training model: {e}")
    return redirect(url_for("index"))

@app.route("/generate", methods=["POST"])
def generate():
    global adapter
    try:
        if adapter is None or not model_trained:
            raise ValueError("Model is not trained.")
        
        sample_size = int(request.form.get("sample_size"))
        synthetic_data = adapter.generate(size=sample_size)
        
        if synthetic_data is None or len(synthetic_data) == 0:
            raise ValueError("No synthetic data generated.")
        
        # Convert to DataFrame for CSV export
        synthetic_df = pd.DataFrame(synthetic_data)
        output = BytesIO()
        synthetic_df.to_csv(output, index=False)
        output.seek(0)
        
        flash("Synthetic data generated successfully!")
        return send_file(output, mimetype="text/csv", as_attachment=True, download_name="synthetic_data.csv")
    except Exception as e:
        flash(f"Error generating synthetic data: {e}")
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from katabatic_logic import load_data, train_model, generate_data

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded files
app.config['ALLOWED_EXTENSIONS'] = {'csv'}  # Restrict to CSV files

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """
    Check if the file has an allowed extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get dataset size and epochs from the request form
    size = int(request.form.get('size', 5))
    epochs = int(request.form.get('epochs', 5))
    
    # Check if a file is uploaded
    if 'file' in request.files:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"[INFO] File saved to {filepath}")
            
            # Load user-provided data
            data = load_data(filepath)
        else:
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed.'}), 400
    else:
        # Fall back to demo data if no file is uploaded
        print("[INFO] No file uploaded. Using demo data.")
        data = load_data()
    
    # Train the model
    model, reliability = train_model(data, epochs=epochs)
    
    # Generate synthetic data
    synthetic_data = generate_data(model, size)
    
    # Convert synthetic data to JSON for UI display
    synthetic_data_dict = synthetic_data.to_dict(orient='records')
    return jsonify({
        'synthetic_data': synthetic_data_dict,
        'reliability': reliability
    })

if __name__ == '__main__':
    app.run(debug=True)
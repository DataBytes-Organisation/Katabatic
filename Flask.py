from flask import Flask, render_template, request, send_file, jsonify, Response
from werkzeug.utils import secure_filename
import os
import tempfile
import time
import pandas as pd
import matplotlib.pyplot as plt
from meg import preprocess_data, build_kge_matrix, train_meg_model, generate_synthetic_data, save_synthetic_data
from ctgan_adapter import CtganAdapter
from ctgan_utils import preprocess_data
from io import BytesIO
import json
from ganblr.models import GANBLR
from ganblrpp_model import GANBLRPP
import matplotlib
matplotlib.use('Agg') 
from sklearn.preprocessing import OrdinalEncoder  # ✅ THIS LINE IS IMPORTANT


UPLOAD_FOLDER = tempfile.gettempdir()
uploaded_df = None
X_train = y_train = X_test = y_test = None
X_train_enc = X_test_enc = encoder = None
model = None
synthetic_df = None


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

@app.route('/')
def home():
    # Render the main HTML template when accessing the root route
    return render_template('index.html')

@app.route('/about')
def about():
    # Render the About Us page from templates folder when the '/about' route is accessed
    return render_template('about.html')

@app.route('/services')
def services():
    # Render the services page from templates folder when the '/services' route is accessed
    return render_template('services.html')

@app.route('/Contact')
def Contact():
    # Render the Contact page from templates folder when the '/Contact' route is accessed
    return render_template('Contact.html')

#--------------------------------------------#
#------------------- GLANBLR ----------------#
#--------------------------------------------#
@app.route('/glanblr')
def glanblr():
    # Render the Glanblr page from templates folder when the '/glanblr' route is accessed
    return render_template('models/glanblr.html')

@app.route('/glanblr/upload', methods=['POST'])
def upload_glanblr():
    global uploaded_df
    if 'file' not in request.files:
        return render_template('models/glanblr.html', output="No file uploaded.")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('models/glanblr.html', output="No selected file.")
    
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    uploaded_df = pd.read_csv(filepath)

    return render_template('models/glanblr.html', output=f"Dataset '{file.filename}' uploaded successfully!")


@app.route('/preprocess', methods=['POST'])
def preprocess():
    global uploaded_df, X_train, y_train, X_test, y_test, X_train_enc, X_test_enc, encoder

    if uploaded_df is None:
        return render_template('models/glanblr.html', output="No dataset uploaded.")

    df = uploaded_df.copy()
    df = df.apply(lambda col: col.str.strip().str.rstrip('.') if col.dtype == "object" else col)

    # Split into train/test (for demo, split manually)
    split = int(0.8 * len(df))
    train_df, test_df = df[:split], df[split:]
    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int)
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    return render_template('models/glanblr.html', output="Data preprocessing completed.")

@app.route('/train', methods=['POST'])
def train_gamblr():
    global model, X_train, y_train

    if X_train is None or y_train is None:
        return render_template('models/glanblr.html', output="Preprocessed data not found.")

    model = GANBLR()
    model.fit(X_train, y_train, k=0, epochs=10, batch_size=64)

    return render_template('models/glanblr.html', output="GANBLR model trained successfully.")

@app.route('/verify', methods=['POST'])
def verify():
    global model, X_train, y_train, synthetic_df

    if model is None or X_train is None or y_train is None:
        return render_template('models/glanblr.html', output="⚠️ Model not trained or data not preprocessed.")

    # Generate synthetic data using GANBLR
    synthetic_data = model.sample(size=1000, verbose=0)
    synthetic_columns = list(X_train.columns) + ['income']
    synthetic_df = pd.DataFrame(synthetic_data, columns=synthetic_columns)

    # Prepare output message and preview table
    real_shape = X_train.assign(income=y_train).shape
    synth_shape = synthetic_df.shape
    preview_html = synthetic_df.head(10).to_html(classes="data", index=False)

    return render_template(
        'models/glanblr.html',
        output=f"✅ Synthetic data generated!<br>Real shape: {real_shape}, Synthetic shape: {synth_shape}",
        preview_table=preview_html
    )

@app.route('/visualize', methods=['POST'])
def visualize():
    global model, X_train, y_train, synthetic_df

    if model is None or X_train is None or y_train is None:
        return render_template('models/glanblr.html', output="⚠️ Please train the model first.")

    # Generate synthetic data
    synthetic_data = model.sample(size=1000, verbose=0)
    synthetic_columns = list(X_train.columns) + ['income']
    synthetic_df = pd.DataFrame(synthetic_data, columns=synthetic_columns)

    # Plot distribution of the first feature
    feature = X_train.columns[0]
    plt.figure(figsize=(6, 4))
    plt.hist(X_train[feature], bins=15, alpha=0.5, label='Real')
    plt.hist(synthetic_df[feature], bins=15, alpha=0.5, label='Synthetic')
    plt.legend()
    plt.title(f'Distribution of {feature}')
    plot_path = os.path.join('static', 'comparison_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return render_template(
    'models/glanblr.html',
    output=f"🧠 Distribution of feature <b>{feature}</b> plotted successfully!",
    preview_table=None,
    show_plot=True  # ✅ Only send this in the /visualize route
)

#--------------------------------------------#
#--------------- GLANBLR END ----------------#
#--------------------------------------------#


#--------------------------------------------#
#------------------- CTGAN ------------------#
#--------------------------------------------#
@app.route('/ctgan')
def ctgan():
    # Render the CTGAN page from templates folder when the '/ctgan' route is accessed
    return render_template('models/CTGAN.html')

dataset = None
categorical_columns = None
synthetic_data = None
training_status = {"status": "idle", "message": ""}  # Global status variable

config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle dataset upload and column extraction.
    """
    global dataset, categorical_columns
    if 'dataset' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['dataset']
    try:
        data_raw = pd.read_csv(file)
        # Preprocess data and detect categorical columns
        dataset, categorical_columns = preprocess_data(data_raw)
        
        # Prepare column information for frontend
        columns_info = []
        for col in dataset.columns:
            columns_info.append({
                'name': col,
                'type': 'categorical' if col in categorical_columns else 'numerical'
            })
        
        return jsonify({
            'columns': dataset.columns.tolist(),
            'columns_info': columns_info,
            'categorical_columns': categorical_columns
        })
    except Exception as e:
        return jsonify({'error': f'Error reading dataset: {str(e)}'}), 500
    
@app.route('/generate', methods=['POST'])
def generate():
    """
    Train the model and generate synthetic data.
    """
    global dataset, synthetic_data, categorical_columns, training_status
    if dataset is None:
        return jsonify({'error': 'No dataset uploaded. Please upload a dataset first.'}), 400

    try:
        selected_columns = request.form.getlist('selected_columns[]')
        num_samples = int(request.form.get('num_samples', 0))
            
        if num_samples <= 0:
            return jsonify({'error': 'Number of synthetic rows must be greater than 0.'}), 400
            
        # Filter dataset to only include selected columns
        data_subset = dataset[selected_columns].copy() if selected_columns else dataset.copy()

        # Step 1: Initializing Model
        training_status['status'] = 'in_progress'
        training_status['message'] = 'Initializing the model...'
        
        # Step 2: Training Model
        training_status['message'] = 'Training the model...'
        model = CtganAdapter(**config["ctgan_params"])
        # No target column specified - train on the entire dataset
        model.fit(data_subset)

        # Step 3: Generating Data
        training_status['message'] = 'Generating synthetic data...'
        synthetic_data = model.generate(num_samples)

        # Convert categorical columns in synthetic data
        for col in synthetic_data.columns:
            if col in categorical_columns:
                synthetic_data[col] = synthetic_data[col].astype('category')

        # Finalizing Training Status
        training_status['status'] = 'completed'
        training_status['message'] = 'Training and data generation completed successfully.'

        # Convert to JSON for frontend display (only first 10 rows for preview)
        preview_data = synthetic_data.head(10)
        synthetic_data_json = preview_data.to_dict(orient='records')
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
def download_CTGAN():
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
#--------------------------------------------#
#----------------- CTGAN END ----------------#
#--------------------------------------------#


#--------------------------------------------#
#------------------- MEG --------------------#
#--------------------------------------------#
@app.route('/meg')
def meg():
    # Render the CTGAN page from templates folder when the '/meg' route is accessed
    return render_template('models/meg.html')

@app.route('/mupload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'status': 'error'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'status': 'error'}), 400

    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Invalid file type. Only CSV files are allowed', 'status': 'error'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV with error handling
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({'error': f'Error reading CSV file: {str(e)}', 'status': 'error'}), 400
        
        if len(df) < 10:
            return jsonify({'error': 'Dataset too small (min 10 rows required)', 'status': 'error'}), 400
        
        rows_to_process = request.form.get('num_rows', type=int, default=None)
        if rows_to_process:
            df = df.head(rows_to_process)

        def generate_events():
            try:
                start_time = time.time()
                
                # Step 1: Preprocess data
                yield "data: Starting data preprocessing...\n\n"
                X_scaled, y, cat_cols, df_full, scaler, encoders = preprocess_data(df.copy())
                target_col = df.columns[-1]
                yield f"data: Preprocessing completed in {time.time() - start_time:.1f} seconds\n\n"
                
                # Step 2: Build KGE Matrix
                yield "data: Building KGE matrix...\n\n"
                kge_start = time.time()
                kge_matrix = build_kge_matrix(df_full, cat_cols)
                yield f"data: KGE Matrix built in {time.time() - kge_start:.1f} seconds\n\n"
                
                # Step 3: Train MEG Model
                yield "data: Training MEG model...\n\n"
                train_start = time.time()
                meg_model = train_meg_model(
                    X_scaled, 
                    kge_matrix, 
                    input_dim=X_scaled.shape[1], 
                    kge_dim=kge_matrix.shape[1],
                    epochs=15
                )
                yield f"data: Model trained in {time.time() - train_start:.1f} seconds\n\n"
                
                # Step 4: Generate Synthetic Data
                yield "data: Generating synthetic data...\n\n"
                gen_start = time.time()
                num_samples = request.form.get('num_samples', default=len(df), type=int)
                synthetic_data = generate_synthetic_data(meg_model, X_scaled, kge_matrix, num_samples)
                yield f"data: Data generated in {time.time() - gen_start:.1f} seconds\n\n"
                
                # Step 5: Save Results
                yield "data: Saving results...\n\n"
                save_start = time.time()
                output_path = save_synthetic_data(synthetic_data, df_full, scaler, encoders, target_col)
                yield f"data: Results saved in {time.time() - save_start:.1f} seconds\n\n"
                
                total_time = time.time() - start_time
                yield f"data: Generation completed in {total_time:.1f} seconds\n\n"
                yield f"data: DONE|{output_path}\n\n"
                
            except Exception as e:
                yield f"data: ERROR|Error during processing: {str(e)}\n\n"

        return Response(generate_events(), mimetype="text/event-stream")

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}', 'status': 'error'}), 500

@app.route('/download')
def download():
    try:
        filepath = request.args.get('path')
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404
            
        return send_file(
            filepath,
            as_attachment=True,
            download_name="synthetic_data.csv",
            mimetype="text/csv"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# Ganblr++ code
@app.route('/ganblrpp')
def ganblrpp():
    # Render the CTGAN page from templates folder when the '/meg' route is accessed
    return render_template('models/ganblrpp.html')

@app.route('/gp_train', methods=['POST'])
def train_ganblrpp():
    try:
        file = request.files['file']
        if not file or file.filename == '':
            return Response("data: ERROR|No file uploaded.\n\n", mimetype='text/event-stream')

        df = pd.read_csv(file)
        if df.shape[0] < 10:
            return Response("data: ERROR|Too few rows (min 10 required).\n\n", mimetype='text/event-stream')

        num_rows = int(request.form.get("num_rows", len(df)))
        num_samples = int(request.form.get("num_samples", 100))
        df = df.head(num_rows)

        model = GANBLRPP()
        X = model.preprocess(df)
        model.train(X)

        synthetic_df = model.generate(num_samples)

        output = BytesIO()
        synthetic_df.to_csv(output, index=False)
        output.seek(0)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "ganblrpp_output.csv")
        with open(temp_path, "wb") as f:
            f.write(output.read())

        return Response(f"data: DONE|{temp_path}\n\n", mimetype='text/event-stream')

    except Exception as e:
        return Response(f"data: ERROR|{str(e)}\n\n", mimetype='text/event-stream')
    


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True ,port= 5008)

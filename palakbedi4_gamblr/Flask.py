from flask import Flask, request, redirect, url_for, render_template
import os
import pandas as pd
import tempfile
from sklearn.preprocessing import OrdinalEncoder  # ✅ THIS LINE IS IMPORTANT
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from ganblr.models import GANBLR
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt




UPLOAD_FOLDER = tempfile.gettempdir()
# Initialize the Flask app

app = Flask(__name__)



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

@app.route('/glanblr')
def glanblr():
    # Render the Glanblr page from templates folder when the '/glanblr' route is accessed
    return render_template('models/glanblr.html')

@app.route('/upload', methods=['POST'])
def upload():
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
def train():
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

@app.route('/ctgan')
def ctgan():
    # Render the CTGAN page from templates folder when the '/ctgan' route is accessed
    return render_template('models/CTGAN.html')

   

@app.route('/meg')
def meg():
    # Render the CTGAN page from templates folder when the '/meg' route is accessed
    return render_template('models/meg.html')

@app.route('/model/<model_name>')
def model_page(model_name):
    # Check if the requested model name is valid
    valid_models = ['glanblr', 'CTGAN', 'meg']
    if model_name in valid_models:
        return render_template(f'models/{model_name}.html', model_name=model_name)
    else:
        # If the model name is not valid, redirect to home or show an error page
        return redirect(url_for('home'))

# Run the Flask app with debugging enabled
if __name__ == '__main__':
    app.run(debug=True, port= 5007)

from flask import Flask, render_template_string, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, BatchNormalization, Input
from keras.optimizers import Adam

app = Flask(__name__)
app.secret_key = 'secret_key'
UPLOAD_FOLDER = "uploaded_datasets"
VISUALIZATION_DIR = "static/visualizations"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Globals for storing datasets
DATASETS = {}
CURRENT_STEP_OUTPUT = None

INDEX_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>GANBLR Interface</title>
    <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
</head>
<body>
    <div class=\"container mt-5\">
        <h1 class=\"text-center\">GANBLR Project Interface</h1>
        <p class=\"text-center\">Upload, preprocess, train, and visualize data with ease.</p>

        <!-- Flash messages -->
        {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
        <div class=\"mt-3\">
            {% for category, message in messages %}
            <div class=\"alert alert-{{ category }}\" role=\"alert\">
                {{ message }}
            </div>
            {% endfor %}
        </div>
        {% endif %}
        {% endwith %}

        <div class=\"row mt-4\">
            <!-- Load Dataset -->
            <div class=\"col-md-4\">
                <form method=\"POST\" action=\"/load_dataset\" enctype=\"multipart/form-data\" class=\"card p-3\">
                    <h5>Step 1: Load Dataset</h5>
                    <input type=\"file\" name=\"datasetFile\" class=\"form-control mb-3\" required>
                    <button type=\"submit\" class=\"btn btn-primary w-100\">Upload Dataset</button>
                </form>
            </div>

            <!-- Preprocess Data -->
            <div class=\"col-md-4\">
                <form method=\"POST\" action=\"/preprocess_data\" class=\"card p-3\">
                    <h5>Step 2: Preprocess Data</h5>
                    <button type=\"submit\" class=\"btn btn-primary w-100\">Preprocess Data</button>
                </form>
            </div>

            <!-- Train Model -->
            <div class=\"col-md-4\">
                <form method=\"POST\" action=\"/train_model\" class=\"card p-3\">
                    <h5>Step 3: Train Model</h5>
                    <button type=\"submit\" class=\"btn btn-primary w-100\">Train Model</button>
                </form>
            </div>
        </div>

        <div class=\"row mt-4\">
            <!-- Verify Data -->
            <div class=\"col-md-4\">
                <form method=\"POST\" action=\"/verify_data\" class=\"card p-3\">
                    <h5>Step 4: Verify Data</h5>
                    <button type=\"submit\" class=\"btn btn-primary w-100\">Verify Data</button>
                </form>
            </div>

            <!-- Generate Visualizations -->
            <div class=\"col-md-4\">
                <form method=\"POST\" action=\"/generate_visualizations\" class=\"card p-3\">
                    <h5>Step 5: Generate Visualizations</h5>
                    <button type=\"submit\" class=\"btn btn-primary w-100\">Generate Visualizations</button>
                </form>
            </div>
        </div>

        <!-- Dynamic Content Section -->
        <div class=\"row mt-4\">
            {% if current_output %}
            <div class=\"col-md-12\">
                <h5>Step Output:</h5>
                <div class=\"table-responsive\">
                    {{ current_output|safe }}
                </div>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>"""

@app.route('/')
def index():
    global CURRENT_STEP_OUTPUT
    return render_template_string(INDEX_HTML, current_output=CURRENT_STEP_OUTPUT)

@app.route('/home')
def home():
    return redirect(url_for('index'))

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    global CURRENT_STEP_OUTPUT
    file = request.files.get('datasetFile')
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        DATASETS['original'] = pd.read_csv(file_path)
        CURRENT_STEP_OUTPUT = DATASETS['original'].head(10).to_html(classes='table table-striped', index=False)
        flash("Dataset loaded successfully!", "success")
    else:
        flash("Please upload a dataset file.", "danger")
    return redirect(url_for('index'))

@app.route('/preprocess_data', methods=['POST'])
def preprocess_data():
    global CURRENT_STEP_OUTPUT
    if 'original' not in DATASETS:
        flash("No dataset loaded. Please load a dataset first.", "danger")
        return redirect(url_for('index'))

    credit_data = DATASETS['original']
    scaler = MinMaxScaler()
    credit_data[['Amount', 'Time']] = scaler.fit_transform(credit_data[['Amount', 'Time']])
    X = credit_data.drop(columns=['Class'])
    y = credit_data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_X_train.csv'), index=False)
    X_test.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_X_test.csv'), index=False)
    y_train.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_y_train.csv'), index=False)
    y_test.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_y_test.csv'), index=False)

    DATASETS['preprocessed'] = credit_data
    CURRENT_STEP_OUTPUT = credit_data.head(10).to_html(classes='table table-striped', index=False)

    flash("Data preprocessing complete.", "success")
    return redirect(url_for('index'))

@app.route('/train_model', methods=['POST'])
def train_model():
    global CURRENT_STEP_OUTPUT
    try:
        # Load training data
        X_train = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_X_train.csv')).to_numpy()
        y_train = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_y_train.csv')).to_numpy().ravel()

        if X_train.size == 0 or y_train.size == 0:
            flash("Training data is empty. Please preprocess the dataset.", "danger")
            return redirect(url_for('index'))

        # GAN parameters
        input_dim = X_train.shape[1]
        latent_dim = 100
        batch_size = 256
        epochs = 1000

        # Build the Generator
        generator = Sequential([
            Input(shape=(latent_dim,)),
            Dense(128),
            LeakyReLU(negative_slope=0.2),
            BatchNormalization(momentum=0.8),
            Dense(256),
            LeakyReLU(negative_slope=0.2),
            BatchNormalization(momentum=0.8),
            Dense(input_dim, activation='tanh')
        ])

        # Build the Discriminator
        discriminator = Sequential([
            Input(shape=(input_dim,)),
            Dense(256),
            LeakyReLU(negative_slope=0.2),
            Dense(128),
            LeakyReLU(negative_slope=0.2),
            Dense(1, activation='sigmoid')
        ])

        # Compile the Discriminator
        discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

        # Build and compile the GAN
        discriminator.trainable = False
        gan = Sequential([generator, discriminator])
        gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

        # Training the GAN
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            try:
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                real_data = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                synthetic_data = generator.predict_on_batch(noise)

                d_loss_real = discriminator.train_on_batch(real_data, real_labels)
                d_loss_fake = discriminator.train_on_batch(synthetic_data, fake_labels)

                noise = np.random.normal(0, 1, (batch_size, latent_dim))
                g_loss = gan.train_on_batch(noise, real_labels)

                if epoch % 100 == 0:
                    print(f"Epoch {epoch}/{epochs} | D Loss Real: {d_loss_real[0]:.4f} | D Loss Fake: {d_loss_fake[0]:.4f} | G Loss: {g_loss:.4f}")

            except Exception as e:
                print(f"Error during training at epoch {epoch}: {e}")
                break

        # Generate synthetic data
        noise = np.random.normal(0, 1, (1000, latent_dim))
        synthetic_data = generator.predict(noise)
        synthetic_df = pd.DataFrame(synthetic_data, columns=[f"Feature_{i}" for i in range(input_dim)])
        synthetic_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_synthetic_data.csv'), index=False)

        CURRENT_STEP_OUTPUT = "<p>Model training complete. Synthetic data generated.</p>"
        flash("Model training complete. Synthetic data generated and saved.", "success")

    except Exception as e:
        flash(f"Error during training: {e}", "danger")

    return redirect(url_for('index'))

@app.route('/verify_data', methods=['POST'])
def verify_data():
    global CURRENT_STEP_OUTPUT
    try:
        synthetic_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_synthetic_data.csv'))
        CURRENT_STEP_OUTPUT = synthetic_data.head(10).to_html(classes='table table-striped', index=False)
        flash("Synthetic data verification complete.", "success")
    except Exception as e:
        flash(f"Error during verification: {e}", "danger")
    return redirect(url_for('index'))

@app.route('/generate_visualizations', methods=['POST'])
def generate_visualizations():
    global CURRENT_STEP_OUTPUT
    try:
        real_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_X_train.csv'))
        synthetic_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'credit_synthetic_data.csv'))

        real_corr = real_data.corr()
        synthetic_corr = synthetic_data.corr()

        real_path = os.path.join(VISUALIZATION_DIR, "real_data_correlation_heatmap.png")
        synthetic_path = os.path.join(VISUALIZATION_DIR, "synthetic_data_correlation_heatmap.png")

        plt.figure(figsize=(10, 8))
        sns.heatmap(real_corr, annot=False, cmap='coolwarm', cbar=True)
        plt.title("Correlation Heatmap - Real Data")
        plt.savefig(real_path)
        plt.close()

        plt.figure(figsize=(10, 8))
        sns.heatmap(synthetic_corr, annot=False, cmap='coolwarm', cbar=True)
        plt.title("Correlation Heatmap - Synthetic Data")
        plt.savefig(synthetic_path)
        plt.close()

        CURRENT_STEP_OUTPUT = f"""
            <h5>Generated Visualizations:</h5>
            <div class='row'>
                <div class='col-md-6'>
                    <h6>Real Data Correlation Heatmap</h6>
                    <img src='/static/visualizations/real_data_correlation_heatmap.png' class='img-fluid' alt='Real Data Heatmap'>
                </div>
                <div class='col-md-6'>
                    <h6>Synthetic Data Correlation Heatmap</h6>
                    <img src='/static/visualizations/synthetic_data_correlation_heatmap.png' class='img-fluid' alt='Synthetic Data Heatmap'>
                </div>
            </div>
        """
        flash("Visualizations generated successfully.", "success")
    except Exception as e:
        flash(f"Error generating visualizations: {e}", "danger")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

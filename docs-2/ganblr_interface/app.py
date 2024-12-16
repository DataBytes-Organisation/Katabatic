from flask import Flask, render_template, request, url_for
import subprocess
import os

app = Flask(__name__)

# Define output directory for visualizations
VISUALIZATION_DIR = "static/visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

@app.route('/')
def index():
    """
    Renders the homepage.
    """
    return render_template('index.html', status="", outputs={})

def run_script(script_name, output_flag=None):
    """
    Helper function to run a script and return its output.
    """
    try:
        command = ["python", script_name]
        if output_flag:
            command.extend(["--output", VISUALIZATION_DIR])
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "output": e.stderr}

@app.route('/run/<action>', methods=['POST'])
def run_action(action):
    """
    Routes actions to their respective scripts and displays outputs.
    """
    outputs = {}
    if action == "load_dataset":
        outputs['load_dataset'] = run_script("load_credit_dataset.py")
    elif action == "preprocess_data":
        outputs['preprocess_data'] = run_script("preprocess_credit_data.py")
    elif action == "train_model":
        outputs['train_model'] = run_script("train_ganblr_credit.py")
    elif action == "verify_data":
        outputs['verify_data'] = run_script("verify_synthetic_data.py")
    elif action == "generate_visualizations":
        outputs['feature_distribution'] = run_script("feature-distribution.py", output_flag=True)
        outputs['correlation_heatmaps'] = run_script("correlation-heat-maps.py", output_flag=True)

        # List generated visualizations
        feature_plots = [f for f in os.listdir(VISUALIZATION_DIR) if f.endswith('_comparison.png')]
        heatmap_plots = [f for f in os.listdir(VISUALIZATION_DIR) if 'heatmap' in f]

        return render_template('visualizations.html',
                               status="Visualizations generated successfully!",
                               feature_plots=feature_plots,
                               heatmap_plots=heatmap_plots,
                               visualization_dir=VISUALIZATION_DIR)

    return render_template('index.html', status="Task completed successfully.", outputs=outputs)

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

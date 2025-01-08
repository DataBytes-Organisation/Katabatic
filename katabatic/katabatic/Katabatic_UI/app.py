import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, render_template, request, jsonify
from katabatic_logic import load_demo_data, train_model, generate_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Input: Accept optional dataset size from UI
    size = int(request.form.get('size', 5))
    demo_data = load_demo_data()
    model = train_model(demo_data)
    synthetic_data = generate_data(model, size)
    # Convert synthetic data to JSON for UI display
    synthetic_data_dict = synthetic_data.to_dict(orient='records')
    return jsonify({'synthetic_data': synthetic_data_dict})

if __name__ == '__main__':
    app.run(debug=True)
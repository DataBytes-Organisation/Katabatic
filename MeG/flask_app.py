from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index_upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # Get dropdown option
    selected_option = request.form.get('option')

    if selected_option == "none":
        return "Nothing is generating!", 400  # Error for "None" selection

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the file to the uploads folder
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Process the CSV file
    data = pd.read_csv(filepath)

    # Render the data in a table
    return render_template(
        'table.html',
        column_names=data.columns.values,
        row_data=list(data.values.tolist()),
        zip=zip
    )


if __name__ == '__main__':
    app.run(debug=True)

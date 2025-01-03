from flask import Flask, render_template

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the main HTML template when accessing the root route
    return render_template('index.html')

# Run the Flask app with debugging enabled
if __name__ == '__main__':
    app.run(debug=True)

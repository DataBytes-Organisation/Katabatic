from flask import Flask, render_template, request, redirect, url_for, flash, session
import os

app = Flask(__name__)
app.secret_key = 'secret_key'

# Route Definitions
@app.route('/')
def home():
    """Render the main homepage."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Render the About Us page."""
    return render_template('about.html')

@app.route('/services')
def services():
    """Render the Services page."""
    return render_template('services.html')

@app.route('/contact')
def contact():
    """Render the Contact Us page."""
    return render_template('contact.html')

@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    """Render the dashboard page."""
    if request.method == 'POST':
        # Placeholder for file upload handling
        return render_template('dashboard.html')
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(debug=True)

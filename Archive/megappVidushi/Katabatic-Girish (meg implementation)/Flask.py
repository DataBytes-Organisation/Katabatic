from flask import Flask, render_template, request, redirect, url_for

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
    app.run(debug=True)

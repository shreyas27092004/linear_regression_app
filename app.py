import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey' # Needed for flashing messages

# --- Custom Linear Regression Model ---
class SimpleLinearRegression:
    """A simple linear regression model built from scratch."""
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        """Fit the model to the data."""
        x_mean = np.mean(X)
        y_mean = np.mean(y)
        
        # Calculate slope (m) and intercept (c)
        # using the formula: m = sum((x - x_mean) * (y - y_mean)) / sum((x - x_mean)^2)
        numerator = np.sum((X - x_mean) * (y - y_mean))
        denominator = np.sum((X - x_mean)**2)
        
        self.slope = numerator / denominator
        self.intercept = y_mean - (self.slope * x_mean)

    def predict(self, X):
        """Make predictions using the fitted model."""
        return self.slope * X + self.intercept

# --- Helper Functions ---
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Redirect to the analysis page with the filename as a parameter
            return redirect(url_for('analyze', filename=filename))

    return render_template('index.html')

@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        flash(f'Error reading CSV file: {e}')
        return redirect(url_for('index'))

    # --- Data Analysis ---
    # For simplicity, we'll assume the first column is the independent variable (X)
    # and the second column is the dependent variable (y).
    if df.shape[1] < 2:
        flash('CSV file must have at least two columns.')
        return redirect(url_for('index'))
        
    X_col, y_col = df.columns[0], df.columns[1]
    X = df[X_col].values
    y = df[y_col].values

    # Fit our custom model
    model = SimpleLinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    # --- Results for Template ---
    results = {
        'filename': filename,
        'num_rows': df.shape[0],
        'num_cols': df.shape[1],
        'x_variable': X_col,
        'y_variable': y_col,
        'slope': f'{model.slope:.4f}',
        'intercept': f'{model.intercept:.4f}',
        'head': df.head().to_html(classes='table-auto w-full text-left', justify='left', border=0)
    }

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X, y=y, label='Actual Data')
    plt.plot(X, predictions, color='red', linewidth=2, label='Regression Line')
    plt.title(f'Linear Regression: {y_col} vs. {X_col}', fontsize=16)
    plt.xlabel(X_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.legend()
    plt.grid(True)
    
    # Ensure the static directory exists
    static_dir = 'static'
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        
    plot_path = os.path.join(static_dir, 'regression_plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    results['plot_url'] = 'static/regression_plot.png'

    return render_template('index.html', results=results)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)

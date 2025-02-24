import os
import csv
import logging
import random
import subprocess
from datetime import datetime
import time
import threading
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
import joblib
import json
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Optional

app = Flask(__name__)
# Use environment variable for secret key, with fallback for local development
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'b8b723e8fac42ffc42ceca66f8f8d2eb')  # Use a secure key in production

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# ---------------------------------------------------------------------
# Load Model (initially)
# ---------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, '..')  # Ensure /app/ for Railway
model_path = os.path.join(base_dir, 'model.pkl')
data_dir = os.path.join(base_dir, 'data')

try:
    os.makedirs(data_dir, exist_ok=True)  # Ensure data directory exists
    model = joblib.load(model_path)
    logger.info("Model loaded successfully from %s", model_path)
except Exception as e:
    model = None
    logger.error("Could not load model from %s. Error: %s. Attempting to retrain if feedback exists...", model_path, e)
    # Attempt to retrain if feedback exists and model fails to load
    feedback_file = os.path.join(data_dir, 'raw', 'feedback.csv')
    if os.path.exists(feedback_file) and pd.read_csv(feedback_file, header=None).shape[0] > 0:
        logger.info("Initiating automatic retraining due to missing model...")
        threading.Thread(target=retrain_model_async, daemon=True).start()

# ---------------------------------------------------------------------
# Training Status Variables
# ---------------------------------------------------------------------
training_status = {
    'is_training': False,
    'progress': 0,
    'estimated_time': "N/A",
    'feedback_count': 0,  # Current queue count from feedback.csv
    'total_feedback_count': 0  # Cumulative count across all time
}

FEEDBACK_THRESHOLD = 50  # Trigger retraining after 50 feedback entries

# Persistent file for cumulative feedback count
CUMULATIVE_FEEDBACK_FILE = os.path.join(data_dir, 'raw', 'cumulative_feedback_count.json')

def load_cumulative_feedback_count() -> int:
    """Load the cumulative feedback count from a persistent file."""
    try:
        if os.path.exists(CUMULATIVE_FEEDBACK_FILE):
            with open(CUMULATIVE_FEEDBACK_FILE, 'r') as f:
                data = json.load(f)
                return data.get('total_feedback_count', 0)
        return 0
    except Exception as e:
        logger.error("Error loading cumulative feedback count: %s", e)
        return 0

def save_cumulative_feedback_count(count: int) -> None:
    """Save the cumulative feedback count to a persistent file."""
    try:
        with open(CUMULATIVE_FEEDBACK_FILE, 'w') as f:
            json.dump({'total_feedback_count': count}, f)
        logger.debug(f"Saved cumulative feedback count: {count}")
    except Exception as e:
        logger.error("Error saving cumulative feedback count: %s", e)

# Initialize cumulative count on app startup
training_status['total_feedback_count'] = load_cumulative_feedback_count()

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def get_feedback_count() -> int:
    """Count the number of unique entries in feedback.csv, ignoring duplicates by message."""
    feedback_file = os.path.join(data_dir, 'raw', 'feedback.csv')
    try:
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file, header=None, names=['label', 'message', 'timestamp'], dtype={'message': str, 'label': str})
            # Ensure message and label columns are strings, handle NaN or None
            df['message'] = df['message'].fillna('').astype(str).str.strip()
            df['label'] = df['label'].fillna('ham').astype(str).str.strip()  # Default to 'ham' for invalid labels
            # Deduplicate by message to prevent counting duplicates
            unique_count = len(df.drop_duplicates(subset=['message']))
            logger.debug(f"Feedback count calculated: {unique_count} unique entries from {feedback_file}")
            return unique_count
        return 0
    except Exception as e:
        logger.error("Error counting feedback: %s", e)
        return 0

def is_duplicate_feedback(message: str) -> bool:
    """Check if the message is already in feedback.csv to prevent duplicates."""
    feedback_file = os.path.join(data_dir, 'raw', 'feedback.csv')
    try:
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file, header=None, names=['label', 'message', 'timestamp'], dtype={'message': str, 'label': str})
            # Ensure message column is strings, handle NaN or None
            df['message'] = df['message'].fillna('').astype(str).str.strip()
            df['label'] = df['label'].fillna('ham').astype(str).str.strip()
            return df['message'].eq(message.strip()).any()
        return False
    except Exception as e:
        logger.error("Error checking for duplicate feedback: %s", e)
        return False  # Default to False to allow submission if check fails

def read_training_progress(progress_file: str) -> tuple[int, str, str]:
    """Read progress from training_progress.json."""
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return data.get('progress', 0), data.get('estimated_time', "N/A"), data.get('status', "running")
        return 0, "N/A", "idle"
    except Exception as e:
        logger.error("Error reading training progress: %s", e)
        return 0, "N/A", "idle"

def retrain_model_async():
    """Run retraining in a background thread with progress updates from train.py, preserving feedback.csv."""
    global model, training_status
    training_status['is_training'] = True
    progress_file = os.path.join(base_dir, 'training_progress.json')
    feedback_file = os.path.join(data_dir, 'raw', 'feedback.csv')
    feedback_backup = os.path.join(data_dir, 'raw', 'feedback_backup.csv')
    
    try:
        # Backup feedback.csv before preprocessing to preserve it
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file, header=None, names=['label', 'message', 'timestamp'], dtype={'message': str, 'label': str})
            df.to_csv(feedback_backup, index=False, header=False)
            logger.info("Backed up feedback.csv to feedback_backup.csv")

        # Ensure data directory and subdirectories exist
        raw_dir = os.path.join(data_dir, 'raw')
        processed_dir = os.path.join(data_dir, 'processed')
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Run preprocessing with absolute paths
        preprocess_script = os.path.join(base_dir, 'src', 'data_processing', 'preprocess.py')
        subprocess.run(["python", preprocess_script], check=True, cwd=base_dir, env=os.environ.copy())
        
        # Run training script and monitor progress
        train_script = os.path.join(base_dir, 'src', 'model', 'train.py')
        process = subprocess.Popen(["python", train_script], cwd=base_dir, env=os.environ.copy())
        
        # Poll progress file while training runs
        while process.poll() is None:
            progress, est_time, status = read_training_progress(progress_file)
            training_status['progress'] = progress
            training_status['estimated_time'] = est_time
            time.sleep(1)  # Check every second
        
        if process.returncode == 0:
            model = joblib.load(model_path)
            logger.info("Retraining completed successfully")
            flash("Model retraining completed successfully!", "success")
            # Restore feedback.csv after retraining to preserve submissions
            if os.path.exists(feedback_backup):
                df_backup = pd.read_csv(feedback_backup, header=None, names=['label', 'message', 'timestamp'], dtype={'message': str, 'label': str})
                df_backup.to_csv(feedback_file, index=False, header=False)
                os.remove(feedback_backup)
                logger.info("Restored feedback.csv from backup after retraining")
        else:
            raise Exception(f"Training script failed with return code {process.returncode}")
    except Exception as e:
        logger.error("Retraining failed: %s", e)
        flash(f"Retraining failed: {e}", "error")
        # Restore feedback.csv if retraining fails
        if os.path.exists(feedback_backup):
            df_backup = pd.read_csv(feedback_backup, header=None, names=['label', 'message', 'timestamp'], dtype={'message': str, 'label': str})
            df_backup.to_csv(feedback_file, index=False, header=False)
            os.remove(feedback_backup)
            logger.info("Restored feedback.csv from backup after retraining failure")
    finally:
        training_status['is_training'] = False
        training_status['progress'] = 0
        training_status['estimated_time'] = "N/A"
        if os.path.exists(progress_file):
            os.remove(progress_file)  # Clean up progress file

# ---------------------------------------------------------------------
# HOME ROUTE
# ---------------------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route for checking spam vs. ham.
    Updates feedback count and triggers retraining if threshold is met.
    """
    prediction = None
    create_message = None

    # Generate CAPTCHA
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    captcha_question = f"What is {a} + {b}?"
    session['captcha_answer'] = str(a + b)

    # Update feedback counts immediately
    training_status['feedback_count'] = get_feedback_count()
    training_status['total_feedback_count'] = load_cumulative_feedback_count()
    logger.debug(f"Initial feedback count for rendering: Current={training_status['feedback_count']}, Total={training_status['total_feedback_count']}")

    if request.method == 'POST':
        user_text = request.form.get('user_text', '').strip()
        if user_text and model is not None:
            prediction = model.predict([user_text])[0]
            create_message = "Your message has been processed."
            logger.info("Message: '%s' => Prediction: %s", user_text, prediction)
        else:
            create_message = "Please enter some text to check or ensure the model is loaded."
            logger.warning("Empty text submitted or model not loaded.")
            flash(create_message, "warning")

    # Check if retraining should be triggered
    if not training_status['is_training'] and training_status['feedback_count'] >= FEEDBACK_THRESHOLD:
        threading.Thread(target=retrain_model_async, daemon=True).start()
        training_status['estimated_time'] = "Calculating..."
        logger.info("Retraining triggered due to feedback threshold reached")

    return render_template(
        'index.html',
        prediction=prediction,
        create_message=create_message,
        captcha_question=captcha_question,
        training_status=training_status
    )

# ---------------------------------------------------------------------
# FEEDBACK ROUTE
# ---------------------------------------------------------------------
@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback and save to feedback.csv, preventing duplicates and updating counts."""
    user_message = request.form.get("message", "").strip()
    correct_label = request.form.get("correct_label", "").strip()
    captcha_response = request.form.get("captcha_response", "").strip()

    if not user_message:
        flash("Feedback submission failed: Message text cannot be empty.", "error")
        logger.error("Feedback submission failed: Empty message.")
        return redirect(url_for("index"))

    expected_captcha = session.get('captcha_answer', "")
    if not expected_captcha or captcha_response != expected_captcha:
        flash("Incorrect CAPTCHA. Please try again.", "error")
        logger.warning("Incorrect CAPTCHA. Expected %s, got %s", expected_captcha, captcha_response)
        return redirect(url_for("index"))

    # Check for duplicate feedback
    if is_duplicate_feedback(user_message):
        flash("This feedback has already been submitted.", "warning")
        logger.warning("Duplicate feedback detected for message: %s", user_message)
        return redirect(url_for("index"))

    feedback_file = os.path.join(data_dir, 'raw', 'feedback.csv')
    os.makedirs(os.path.dirname(feedback_file), exist_ok=True)

    # Ensure file exists with proper header if new
    if not os.path.exists(feedback_file):
        with open(feedback_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "message", "timestamp"])

    # Append feedback to CSV
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(feedback_file, "a", newline="", encoding="utf-8", errors='replace') as f:
            writer = csv.writer(f)
            writer.writerow([correct_label, user_message, timestamp])
        flash("Feedback submitted successfully!", "success")
        logger.info("Feedback added -> Label: %s, Message: %s, Timestamp: %s", correct_label, user_message, timestamp)
        # Update feedback counts after submission
        training_status['feedback_count'] = get_feedback_count()
        current_cumulative = load_cumulative_feedback_count()
        if not is_duplicate_feedback(user_message, include_history=True):  # Custom check for history
            training_status['total_feedback_count'] = current_cumulative + 1
            save_cumulative_feedback_count(training_status['total_feedback_count'])
        logger.debug(f"Updated feedback counts: Current={training_status['feedback_count']}, Total={training_status['total_feedback_count']}")
    except Exception as e:
        flash("Error saving feedback. Please try again later.", "error")
        logger.error("Error saving feedback: %s", e)

    return redirect(url_for("index"))

# ---------------------------------------------------------------------
# FEEDBACK COUNT ROUTE (for AJAX updates)
# ---------------------------------------------------------------------
@app.route('/feedback_count', methods=['GET'])
def feedback_count():
    """Return the current feedback counts as JSON for UI updates."""
    training_status['feedback_count'] = get_feedback_count()
    training_status['total_feedback_count'] = load_cumulative_feedback_count()
    logger.debug(f"Returning feedback counts: Current={training_status['feedback_count']}, Total={training_status['total_feedback_count']}")
    return jsonify({
        'feedback_count': training_status['feedback_count'],
        'total_feedback_count': training_status['total_feedback_count'],
        'status': 'success'
    }), 200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',  # Allow cross-origin requests for Railway
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Accept'
    }

# ---------------------------------------------------------------------
# OPTIONS Route for CORS Preflight (Optional, for stricter CORS handling)
# ---------------------------------------------------------------------
@app.route('/feedback_count', methods=['OPTIONS'])
def feedback_count_options():
    """Handle CORS preflight requests for /feedback_count."""
    return jsonify({}), 200, {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type, Accept'
    }

# ---------------------------------------------------------------------
# RETRAIN ROUTE (Manual Trigger)
# ---------------------------------------------------------------------
@app.route('/retrain', methods=['GET'])
def retrain():
    """Manual retraining route for testing or admin use."""
    if not training_status['is_training']:
        threading.Thread(target=retrain_model_async, daemon=True).start()
        flash("Manual retraining triggered!", "success")
        logger.info("Manual retraining initiated")
    else:
        flash("Retraining is already in progress.", "warning")
        logger.warning("Retraining already in progress")
    return redirect(url_for("index"))

# Modified is_duplicate_feedback to include historical check
def is_duplicate_feedback(message: str, include_history: bool = False) -> bool:
    """Check if the message is already in feedback.csv or historical backup, optionally."""
    feedback_file = os.path.join(data_dir, 'raw', 'feedback.csv')
    feedback_backup = os.path.join(data_dir, 'raw', 'feedback_backup.csv')
    try:
        is_duplicate = False
        if os.path.exists(feedback_file):
            df = pd.read_csv(feedback_file, header=None, names=['label', 'message', 'timestamp'], dtype={'message': str, 'label': str})
            df['message'] = df['message'].fillna('').astype(str).str.strip()
            df['label'] = df['label'].fillna('ham').astype(str).str.strip()
            is_duplicate |= df['message'].eq(message.strip()).any()
        if include_history and os.path.exists(feedback_backup):
            df_backup = pd.read_csv(feedback_backup, header=None, names=['label', 'message', 'timestamp'], dtype={'message': str, 'label': str})
            df_backup['message'] = df_backup['message'].fillna('').astype(str).str.strip()
            df_backup['label'] = df_backup['label'].fillna('ham').astype(str).str.strip()
            is_duplicate |= df_backup['message'].eq(message.strip()).any()
        return is_duplicate
    except Exception as e:
        logger.error("Error checking for duplicate feedback: %s", e)
        return False  # Default to False to allow submission if check fails

# ---------------------------------------------------------------------
# Run the Flask App
# ---------------------------------------------------------------------
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    port = int(os.environ.get('PORT', 8080))  # Use 8080 as default for Railway
    app.run(debug=False, host="0.0.0.0", port=port)
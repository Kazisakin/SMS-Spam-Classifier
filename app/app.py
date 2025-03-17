import os
import csv
import logging
import random
import subprocess
from datetime import datetime
import time
import threading
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash, session, send_file, jsonify
import joblib
import json

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'b8b723e8fac42ffc42ceca66f8f8d2eb')  # Secure key from env or fallback
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Logging Configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Detailed tracing
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Paths (relative and configurable for Railway/Docker)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)  # Project root (SMS-SPAM-CLASSIFIER/)
data_dir = os.environ.get('DATA_DIR', os.path.join(current_dir, 'data'))  # Use volume or local fallback
model_path = os.path.join(root_dir, 'model.pkl')  # Model in root directory
FEEDBACK_FILE = os.path.join(data_dir, 'feedback.csv')
FALLBACK_FEEDBACK_FILE = os.path.join(current_dir, 'feedback_fallback.csv')
CUMULATIVE_FEEDBACK_FILE = os.path.join(data_dir, 'cumulative_feedback_count.json')
HISTORY_FILE = os.path.join(data_dir, 'user_history.csv')

# Load Model
try:
    os.makedirs(data_dir, exist_ok=True)
    model = joblib.load(model_path)
    logger.info("Model loaded from %s", model_path)
except Exception as e:
    model = None
    logger.error("Could not load model from %s. Error: %s", model_path, e)

# Training Status (defined but not initialized yet)
training_status = {
    'is_training': False,
    'progress': 0,
    'estimated_time': "N/A",
    'feedback_count': 0,
    'total_feedback_count': 0
}
FEEDBACK_THRESHOLD = 50

# Utility Functions (defined before use)
def get_feedback_count() -> int:
    try:
        if os.path.exists(FEEDBACK_FILE):
            df = pd.read_csv(FEEDBACK_FILE, names=['label', 'message', 'timestamp'], header=0 if 'label' in pd.read_csv(FEEDBACK_FILE, nrows=1).columns else None)
            unique_count = len(df.drop_duplicates(subset=['message']))
            logger.debug(f"Feedback count: {unique_count} unique entries in {FEEDBACK_FILE}")
            return unique_count
        logger.debug(f"No feedback file at {FEEDBACK_FILE}")
        return 0
    except Exception as e:
        logger.error("Error counting feedback: %s", e)
        return 0

def get_feedback_stats():
    try:
        if os.path.exists(FEEDBACK_FILE):
            df = pd.read_csv(FEEDBACK_FILE, names=['label', 'message', 'timestamp'], header=0 if 'label' in pd.read_csv(FEEDBACK_FILE, nrows=1).columns else None)
            df = df.dropna(how='all')
            total = len(df)
            if total == 0:
                return {'spam_percent': 0, 'ham_percent': 0}
            spam_count = len(df[df['label'] == 'spam'])
            ham_count = len(df[df['label'] == 'ham'])
            return {
                'spam_percent': round((spam_count / total) * 100, 2) if total > 0 else 0,
                'ham_percent': round((ham_count / total) * 100, 2) if total > 0 else 0
            }
        return {'spam_percent': 0, 'ham_percent': 0}
    except Exception as e:
        logger.error("Error calculating feedback stats: %s", e)
        return {'spam_percent': 0, 'ham_percent': 0}

def is_duplicate_feedback(message: str) -> bool:
    try:
        if os.path.exists(FEEDBACK_FILE):
            df = pd.read_csv(FEEDBACK_FILE, names=['label', 'message', 'timestamp'], header=0 if 'label' in pd.read_csv(FEEDBACK_FILE, nrows=1).columns else None)
            df['message'] = df['message'].fillna('').astype(str).str.strip()
            is_duplicate = message.strip() in df['message'].values
            logger.debug(f"Checking duplicate: '{message}' -> {is_duplicate}")
            return is_duplicate
        return False
    except Exception as e:
        logger.error("Error checking duplicate feedback: %s", e)
        return False

def get_user_history():
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            return df.tail(10).to_dict('records')
        logger.debug(f"No user history file at {HISTORY_FILE}")
        return []
    except Exception as e:
        logger.error("Error reading user history: %s", e)
        return []

def save_user_history(user_text, prediction, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = [user_text, prediction, confidence, timestamp]
    file_exists = os.path.exists(HISTORY_FILE)
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['message', 'prediction', 'confidence', 'timestamp'])
            writer.writerow(entry)
        logger.debug(f"Saved history entry to {HISTORY_FILE}: {entry}")
    except Exception as e:
        logger.error("Error saving user history: %s", e)

def load_cumulative_feedback_count() -> int:
    try:
        if os.path.exists(CUMULATIVE_FEEDBACK_FILE):
            with open(CUMULATIVE_FEEDBACK_FILE, 'r') as f:
                data = json.load(f)
                return data.get('total_feedback_count', 0)
        logger.debug(f"No cumulative feedback file at {CUMULATIVE_FEEDBACK_FILE}")
        return 0
    except Exception as e:
        logger.error("Error loading cumulative feedback count: %s", e)
        return 0

def save_cumulative_feedback_count(count: int) -> None:
    try:
        os.makedirs(os.path.dirname(CUMULATIVE_FEEDBACK_FILE), exist_ok=True)
        with open(CUMULATIVE_FEEDBACK_FILE, 'w') as f:
            json.dump({'total_feedback_count': count}, f)
        logger.debug(f"Saved cumulative feedback count: {count} to {CUMULATIVE_FEEDBACK_FILE}")
    except Exception as e:
        logger.error("Error saving cumulative feedback count: %s", e)

def initialize_feedback_counts():
    feedback_count = get_feedback_count()
    total_count = load_cumulative_feedback_count()
    if total_count < feedback_count:  # Sync if file is outdated
        logger.debug(f"Syncing total feedback count to {feedback_count}")
        save_cumulative_feedback_count(feedback_count)
        total_count = feedback_count
    return feedback_count, total_count

def read_training_progress(progress_file: str) -> tuple[int, str]:
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                data = json.load(f)
                return data.get('progress', 0), data.get('estimated_time', "N/A")
        return 0, "N/A"
    except Exception as e:
        logger.error("Error reading training progress: %s", e)
        return 0, "N/A"

def retrain_model_async():
    global model, training_status
    training_status['is_training'] = True
    progress_file = os.path.join(data_dir, 'training_progress.json')
    feedback_backup = os.path.join(data_dir, 'feedback_backup.csv')
    try:
        if os.path.exists(FEEDBACK_FILE):
            df = pd.read_csv(FEEDBACK_FILE, names=['label', 'message', 'timestamp'], header=0 if 'label' in pd.read_csv(FEEDBACK_FILE, nrows=1).columns else None)
            df.to_csv(feedback_backup, index=False, header=False)
            logger.info("Backed up feedback.csv to %s", feedback_backup)
        
        preprocess_script = os.path.join(root_dir, 'src', 'data_processing', 'preprocess.py')
        train_script = os.path.join(root_dir, 'src', 'model', 'train.py')
        
        if not os.path.exists(preprocess_script) or not os.path.exists(train_script):
            raise FileNotFoundError("Retraining scripts not found in deployment environment")
        
        subprocess.run(["python", preprocess_script], check=True, cwd=root_dir)
        process = subprocess.Popen(["python", train_script], cwd=root_dir)
        start_time = time.time()
        while process.poll() is None:
            progress, est_time = read_training_progress(progress_file)
            if progress == 0 and est_time == "N/A":
                elapsed = time.time() - start_time
                progress = min(90, int(elapsed / 10 * 100))
                est_time = f"{max(0, 10 - int(elapsed))}s"
            training_status['progress'] = progress
            training_status['estimated_time'] = est_time
            time.sleep(1)
        if process.returncode == 0:
            model = joblib.load(model_path)
            logger.info("Retraining completed successfully")
            flash("Model retraining completed successfully!", "success")
            if os.path.exists(feedback_backup):
                df_backup = pd.read_csv(feedback_backup, names=['label', 'message', 'timestamp'], header=0)
                df_backup.to_csv(FEEDBACK_FILE, index=False, header=False)
                os.remove(feedback_backup)
        else:
            raise Exception(f"Training script failed with return code {process.returncode}")
    except Exception as e:
        logger.error("Retraining failed: %s", e)
        flash(f"Retraining failed: {e}", "error")
        if os.path.exists(feedback_backup):
            df_backup = pd.read_csv(feedback_backup, names=['label', 'message', 'timestamp'], header=0)
            df_backup.to_csv(FEEDBACK_FILE, index=False, header=False)
            os.remove(feedback_backup)
    finally:
        training_status['is_training'] = False
        training_status['progress'] = 0
        training_status['estimated_time'] = "N/A"
        if os.path.exists(progress_file):
            os.remove(progress_file)

# Initialize training_status after all utility functions are defined
training_status['feedback_count'], training_status['total_feedback_count'] = initialize_feedback_counts()

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    explanation = None
    create_message = None
    threshold = session.get('threshold', 0.5)
    feedback_stats = get_feedback_stats()
    user_history = get_user_history()
    a, b = random.randint(1, 10), random.randint(1, 10)
    captcha_question = f"What is {a} + {b}?"
    session['captcha_answer'] = str(a + b)
    training_status['feedback_count'] = get_feedback_count()

    if request.method == 'POST':
        user_text = request.form.get('user_text', '').strip()
        threshold = float(request.form.get('threshold', 0.5))
        session['threshold'] = threshold
        if user_text and model:
            probabilities = model.predict_proba([user_text])[0]
            confidence = round(max(probabilities) * 100, 2)
            prediction = 'spam' if probabilities[1] > threshold else 'ham'
            keywords = ['win', 'free', 'prize']
            explanation = "High spam probability due to keywords: " + ", ".join([kw for kw in keywords if kw in user_text.lower()]) if any(kw in user_text.lower() for kw in keywords) else "No strong spam indicators."
            save_user_history(user_text, prediction, confidence)
            create_message = "Your message has been processed."
            logger.info("Message: '%s' => Prediction: %s, Confidence: %.2f%%, Threshold: %.2f", user_text, prediction, confidence, threshold)
        else:
            create_message = "Please enter some text to check or model not loaded."
            logger.warning("Empty text submitted or model not loaded.")

    if not training_status['is_training'] and training_status['feedback_count'] >= FEEDBACK_THRESHOLD:
        threading.Thread(target=retrain_model_async, daemon=True).start()
        training_status['estimated_time'] = "Calculating..."

    return render_template(
        'index.html',
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        create_message=create_message,
        captcha_question=captcha_question,
        training_status=training_status,
        feedback_stats=feedback_stats,
        user_history=user_history,
        threshold=threshold,
        FEEDBACK_THRESHOLD=FEEDBACK_THRESHOLD
    )

@app.route('/feedback', methods=['POST'])
def feedback():
    user_message = request.form.get("message", "").strip()
    correct_label = request.form.get("correct_label", "").strip()
    captcha_response = request.form.get("captcha_response", "").strip()
    
    logger.debug("Feedback route called with: message='%s', label='%s', captcha='%s'", user_message, correct_label, captcha_response)
    
    if not user_message:
        flash("Feedback submission failed: Message text cannot be empty.", "error")
        logger.error("Feedback submission failed: Empty message.")
        return redirect(url_for("index"))
    
    expected_captcha = session.get('captcha_answer', "")
    if captcha_response != expected_captcha:
        flash("Incorrect CAPTCHA. Please try again.", "error")
        logger.warning("Incorrect CAPTCHA. Expected %s, got %s", expected_captcha, captcha_response)
        return redirect(url_for("index"))
    
    if is_duplicate_feedback(user_message):
        flash("This feedback has already been submitted.", "warning")
        logger.warning("Duplicate feedback detected for message: %s", user_message)
        return redirect(url_for("index"))
    
    target_file = FEEDBACK_FILE
    try:
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        logger.debug(f"Attempting to write feedback to: {FEEDBACK_FILE}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.exists(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["label", "message", "timestamp"])
                logger.debug("Created new feedback.csv with headers at %s", FEEDBACK_FILE)
            writer.writerow([correct_label, user_message, timestamp])
            logger.debug("Wrote feedback to %s: %s, %s, %s", FEEDBACK_FILE, correct_label, user_message, timestamp)
    except Exception as e:
        logger.error("Failed to write to %s: %s", FEEDBACK_FILE, e)
        target_file = FALLBACK_FEEDBACK_FILE
        try:
            logger.debug(f"Falling back to write feedback to: {FALLBACK_FEEDBACK_FILE}")
            file_exists = os.path.exists(FALLBACK_FEEDBACK_FILE)
            with open(FALLBACK_FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["label", "message", "timestamp"])
                    logger.debug("Created new feedback_fallback.csv with headers at %s", FALLBACK_FEEDBACK_FILE)
                writer.writerow([correct_label, user_message, timestamp])
                logger.debug("Wrote feedback to %s: %s, %s, %s", FALLBACK_FEEDBACK_FILE, correct_label, user_message, timestamp)
        except Exception as e2:
            logger.error("Failed to write to fallback %s: %s", FALLBACK_FEEDBACK_FILE, e2)
            flash("Error saving feedback to both primary and fallback locations.", "error")
            return redirect(url_for("index"))
    
    flash("Feedback submitted successfully!", "success")
    logger.info("Feedback added -> Label: %s, Message: %s, Timestamp: %s to %s", correct_label, user_message, timestamp, target_file)
    training_status['feedback_count'] = get_feedback_count()
    training_status['total_feedback_count'] += 1
    save_cumulative_feedback_count(training_status['total_feedback_count'])
    return redirect(url_for("index"))

@app.route('/export_feedback')
def export_feedback():
    if os.path.exists(FEEDBACK_FILE):
        return send_file(FEEDBACK_FILE, as_attachment=True, download_name=f"feedback_{datetime.now().strftime('%Y-%m-%d')}.csv")
    elif os.path.exists(FALLBACK_FEEDBACK_FILE):
        return send_file(FALLBACK_FEEDBACK_FILE, as_attachment=True, download_name=f"feedback_fallback_{datetime.now().strftime('%Y-%m-%d')}.csv")
    flash("No feedback data available to export.", "warning")
    return redirect(url_for("index"))

@app.route('/retrain', methods=['GET'])
def retrain():
    if not training_status['is_training']:
        threading.Thread(target=retrain_model_async, daemon=True).start()
        flash("Manual retraining triggered!", "success")
    else:
        flash("Retraining is already in progress.", "warning")
    return redirect(url_for("index"))

@app.route('/feedback_count', methods=['GET'])
def feedback_count():
    training_status['feedback_count'] = get_feedback_count()
    return jsonify({
        'feedback_count': training_status['feedback_count'],
        'total_feedback_count': training_status['total_feedback_count'],
        'status': 'success'
    }), 200

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    port = int(os.environ.get('PORT', 8080))  # Railway sets PORT
    app.run(debug=False, host="0.0.0.0", port=port)  # Debug off for production
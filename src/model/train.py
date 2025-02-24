import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import json
import time
import math

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

def load_training_data(train_path):
    """
    Load training data from a CSV file, handling gzip compression if applicable.
    """
    try:
        # Check if the file exists before attempting to load
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"The file {train_path} does not exist.")
        
        # Check if the file is compressed (ends with .gz)
        if train_path.endswith('.gz'):
            df = pd.read_csv(train_path, compression='gzip')
        else:
            df = pd.read_csv(train_path)
        df['message'] = df['message'].fillna('').astype(str)
        df['label'] = df['label'].fillna('ham').astype(str).str.strip()  # Ensure label is a string, default to 'ham'
        logger.info(f"Training data loaded from {train_path} with shape {df.shape}")
        logger.info("Label distribution:\n%s", df['label'].value_counts())
        return df
    except Exception as e:
        logger.error(f"Unable to load training data from {train_path}. Exception: {e}")
        raise e

def update_progress(progress_file, progress, estimated_time, status="running"):
    """Write training progress to a JSON file."""
    try:
        with open(progress_file, 'w') as f:
            json.dump({
                'progress': progress,
                'estimated_time': estimated_time,
                'status': status
            }, f)
        logger.debug(f"Updated progress to {progress}% in {progress_file}")
    except Exception as e:
        logger.error(f"Error updating progress file {progress_file}: {e}")

def train_model(train_data, progress_file):
    """
    Train the model with progress updates, using TfidfVectorizer and LogisticRegression.
    """
    logger.info("Starting training pipeline...")
    start_time = time.time()
    total_steps = 3  # Data prep, GridSearchCV, Final fit

    # Step 1: Data preparation (0-20%)
    update_progress(progress_file, 10, "Calculating...", "running")
    X = train_data['message']
    y = train_data['label']
    logger.debug("Data prepared for training.")

    # Step 2: GridSearchCV (20-80%)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
    ])
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__min_df': [1, 2],
        'tfidf__max_df': [0.85, 0.95],
        'tfidf__sublinear_tf': [True, False],
        'clf__C': [0.1, 1, 10],
        'clf__penalty': ['l1', 'l2'],
        'clf__class_weight': [None, 'balanced']
    }
    grid_search = GridSearchCV(
        pipeline,
        parameters,
        cv=5,
        n_jobs=-1,
        verbose=2,
        scoring='f1_macro'
    )
    
    # Simulate progress during GridSearchCV
    update_progress(progress_file, 20, "Calculating...", "running")
    grid_search.fit(X, y)
    
    # Estimate time based on GridSearchCV duration
    elapsed_time = time.time() - start_time
    total_estimated_time = elapsed_time / 0.8 * 1.0  # Extrapolate to 100%
    remaining_time = max(0, total_estimated_time - elapsed_time)
    update_progress(progress_file, 80, f"{math.ceil(remaining_time)} seconds remaining", "running")
    
    logger.info("Grid search complete.")
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validated F1 score: {grid_search.best_score_:.4f}")

    # Step 3: Final model (80-100%)
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)  # Final fit with best parameters
    update_progress(progress_file, 100, "0 seconds remaining", "complete")
    
    # Evaluate on training set (debug)
    from sklearn.metrics import f1_score, classification_report
    train_preds = best_model.predict(X)
    f1_train = f1_score(y, train_preds, average='macro')
    logger.info(f"F1 score on training data (macro): {f1_train:.4f}")
    logger.info("Classification report (training data):\n" + classification_report(y, train_preds))
    
    logger.info("Model training complete.")
    return best_model

def save_model(model, model_path):
    """
    Save the trained model to a file.
    Does not modify feedback.csv or related files.
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")
        raise e

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', '..')
    train_path = os.path.join(base_dir, 'data', 'processed', 'train.csv.gz')  # Explicitly use .gz
    model_path = os.path.join(base_dir, 'model.pkl')
    progress_file = os.path.join(base_dir, 'training_progress.json')
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 1) Load training data (only from processed data, not feedback.csv)
    train_data = load_training_data(train_path)
    
    # 2) Train the model with progress updates, ensuring no interaction with feedback.csv
    model = train_model(train_data, progress_file)
    
    # 3) Save the final model, ensuring no impact on feedback.csv
    save_model(model, model_path)
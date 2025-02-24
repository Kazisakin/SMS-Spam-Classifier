# src/model/predict.py

import joblib
import os

def load_model(model_path):
    return joblib.load(model_path)

def predict(text, model):
    """
    Predict whether the given text is spam or ham.
    """
    prediction = model.predict([text])
    return prediction[0]

if __name__ == "__main__":
    model_path = os.path.join('..', '..', 'model.pkl')
    model = load_model(model_path)
    
    sample_text = "Congratulations! You've won a free ticket."
    result = predict(sample_text, model)
    print("Prediction:", result)

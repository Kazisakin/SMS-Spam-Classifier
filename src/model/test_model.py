import joblib
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, '..', '..', 'model.pkl')
test_path = os.path.join(script_dir, '..', '..', 'data', 'processed', 'test.csv')

# Load the newly trained model
model = joblib.load(model_path)

# Load the test dataset
df_test = pd.read_csv(test_path)

if 'message' in df_test.columns and 'label' in df_test.columns:
    X_test = df_test['message'].fillna('')
    y_test = df_test['label'].fillna('')
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"Accuracy on test set: {acc:.4f}")
    print(f"F1 Score (macro) on test set: {f1:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred))
else:
    print("ERROR: test.csv does not have the required columns 'message' and 'label'.")

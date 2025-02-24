📧 Email Spam Classifier 🚫
A machine learning project to detect spam emails or SMS messages using the public SMS Spam Collection dataset.
It provides a Flask web app where users can check if a message is spam or ham, submit feedback, and optionally retrain the model.

🔗 Live Demo: Try it here

✨ Features
✅ Automated Preprocessing: Merges raw data and user feedback into training & test sets.
✅ Machine Learning Model: Uses scikit-learn (Logistic Regression + Tfidf Vectorizer) with hyperparameter tuning.
✅ Web Interface: A Flask-based web app for message classification and user feedback submission.
✅ Continuous Learning: A /retrain endpoint integrates new feedback into the dataset and updates the model.


Email-Spam-Classifier/  
├── .venv/  
├── app/  
│   ├── static/  
│   │   └── style.css  
│   ├── templates/  
│   │   └── index.html  
│   └── app.py  
├── data/  
│   ├── processed/  
│   │   ├── test.csv  
│   │   └── train.csv  
│   └── raw/  
│       ├── dataset.csv  
│       └── feedback.csv  
├── notebooks/  
│   └── EDA.ipynb  
├── src/  
│   ├── data_processing/  
│   │   ├── __init__.py  
│   │   └── preprocess.py  
│   ├── model/  
│   │   ├── __init__.py  
│   │   ├── train.py  
│   │   ├── test_model.py  
│   │   └── predict.py  
│   └── utils/  
│       ├── __init__.py  
│       └── helper.py  
├── model.pkl  
├── requirements.txt  
├── runtime.txt  
└── README.md  


⚙️ Setup Instructions
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/Email-Spam-Classifier.git
cd Email-Spam-Classifier
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Preprocess the Data
bash
Copy
Edit
python src/data_processing/preprocess.py
4️⃣ Train the Model
bash
Copy
Edit
python src/model/train.py
5️⃣ Run the Web App
bash
Copy
Edit
python app/app.py
The Flask server should start on http://127.0.0.1:5000/ by default.

🚀 Live Demo
Try it now

Once loaded:

1️⃣ Enter an SMS or email text to check if it’s Spam or Ham.
2️⃣ Submit feedback if you disagree with the classification (updates feedback.csv).

🔧 Retraining (Optional)
To include new feedback:

Visit /retrain or click the "Retrain Model" button in the UI.
The app merges new feedback, re-trains the model, evaluates its performance, and reloads the updated model.
🏷 License
You may include any license here (MIT, Apache, GPL).

Made with ❤️ using Flask + scikit-learn
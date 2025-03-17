# SMS Spam Classifier

<div>
  <p>This project is a web-based SMS spam classifier built with Flask and deployed using Docker. It leverages a machine learning model to categorize messages as spam or ham, offers user feedback to enhance accuracy, and supports model retraining. Designed for ease of use, it’s containerized for local testing and cloud deployment. This guide walks you through setup, running, and deployment.</p>
</div>

## Features
<ul>
  <li><b>Spam/Ham Detection</b>: Classifies messages with confidence scores.</li>
  <li><b>User Feedback</b>: Collects corrections from users to improve the model.</li>
  <li><b>Model Retraining</b>: Updates the model when feedback reaches a set threshold (50 entries).</li>
  <li><b>Interaction History</b>: Tracks the last 10 user submissions.</li>
  <li><b>CAPTCHA Protection</b>: Adds a simple math challenge to block automated feedback.</li>
  <li><b>Data Persistence</b>: Stores feedback and history using volumes.</li>
</ul>

## Prerequisites
<p>You’ll need these tools to get started:</p>
<ul>
  <li><b>Docker</b>: For containerized execution.</li>
  <li><b>Python 3.11</b>: Optional, for non-Docker runs.</li>
  <li><b>Git</b>: To clone the project repository.</li>
  <li><b>Cloud Platform Account</b>: For deployment (e.g., Railway, optional).</li>
</ul>

## Live Demo
<div>
  <p>Check out the live version of the app here: <a href="https://sms-spam-classifier-production-0e7e.up.railway.app/">https://sms-spam-classifier-production-0e7e.up.railway.app/</a>. Try classifying messages and submitting feedback to see it in action!</p>
</div>

## Project Structure
<p>The project is organized as follows:</p>
<pre>
SMS-SPAM-CLASSIFIER
├── app
│   ├── static
│   │   ├── script.js         <i># Client-side JavaScript for UI</i>
│   │   ├── style.css         <i># CSS for the web interface</i>
│   ├── templates
│   │   ├── index.html        <i># Main UI template</i>
│   ├── app.py                <i># Flask application logic</i>
├── data
│   ├── processed
│   │   ├── test.csv          <i># Processed test data</i>
│   │   ├── train.csv         <i># Processed training data</i>
│   ├── raw
│   │   ├── 20021010_easy_ham.tar.bz2  <i># Raw easy ham dataset</i>
│   │   ├── 20021010_hard_ham.tar.bz2  <i># Raw hard ham dataset</i>
│   │   ├── 20021010_spam.tar.bz2      <i># Raw spam dataset</i>
│   │   ├── 20030228_easy_ham_2.tar.bz2 <i># Additional easy ham</i>
│   │   ├── 20030228_hard_ham.tar.bz2   <i># Additional hard ham</i>
│   │   ├── 20030228_spam_2.tar.bz2     <i># Additional spam</i>
│   │   ├── 20050311_spam_2.tar.bz2     <i># More spam data</i>
│   │   ├── Dataset_5971.csv           <i># Custom dataset</i>
│   │   ├── feedback.csv              <i># User feedback (runtime)</i>
│   │   ├── SMSSpamCollection.csv     <i># SMS spam collection</i>
│   │   ├── spam_ham_dataset.csv      <i># Mixed spam/ham data</i>
│   │   ├── spam.csv                 <i># Spam dataset</i>
│   │   ├── user_history.csv         <i># User history (runtime)</i>
│   │   ├── cumulative_feedback.csv  <i># Feedback count (runtime)</i>
├── notebooks
│   ├── EDA.ipynb             <i># Exploratory Data Analysis</i>
├── src
│   ├── data_processing
│   │   ├── __init__.py       <i># Package initialization</i>
│   │   ├── preprocess.py     <i># Data preprocessing script</i>
│   ├── model
│   │   ├── __init__.py       <i># Package initialization</i>
│   │   ├── predict.py        <i># Prediction utilities</i>
│   │   ├── test_model.py     <i># Model testing script</i>
│   │   ├── train.py          <i># Model training script</i>
│   ├── utils
│   │   ├── __init__.py       <i># Package initialization</i>
│   │   ├── helper.py         <i># Helper functions</i>
├── .gitignore                <i># Git ignore file</i>
├── Dockerfile                <i># Docker build configuration</i>
├── model.pkl                 <i># Pre-trained ML model</i>
├── nixpacks.toml             <i># Nixpacks config (optional)</i>
├── Procfile                  <i># Cloud process file (non-Docker)</i>
├── README.md                 <i># This documentation</i>
├── requirements.txt          <i># Python dependencies</i>
├── runtime.txt               <i># Python runtime version (optional)</i>
</pre>

## How to Run Locally

### Using Docker (Recommended)
<div>
  <ol>
    <li><b>Clone the Repository</b>:<br>
      <code>git clone [repository-url]</code><br>
      <code>cd sms-spam-classifier</code><br>
      <i>Replace [repository-url] with the project’s Git URL.</i>
    </li>
    <li><b>Build the Docker Image</b>:<br>
      <code>docker build -t sms-spam-classifier:latest .</code>
    </li>
    <li><b>Run the Container</b>:<br>
      <code>docker run -p 8080:8080 -e FLASK_SECRET_KEY=[your-secret-key] -v spam-classifier-data:/data sms-spam-classifier:latest</code><br>
      <ul>
        <li><code>-p 8080:8080</code>: Maps port 8080 to your machine.</li>
        <li><code>-e FLASK_SECRET_KEY</code>: Sets a secure key (generate your own).</li>
        <li><code>-v spam-classifier-data:/data</code>: Creates a volume for data persistence.</li>
      </ul>
    </li>
    <li><b>Access the App</b>:<br>
      Visit <a href="http://localhost:8080">http://localhost:8080</a> in your browser.
    </li>
  </ol>
</div>

### Without Docker
<div>
  <ol>
    <li><b>Install Dependencies</b>:<br>
      <code>pip install -r requirements.txt</code><br>
      <i>Requires Python 3.11 installed.</i>
    </li>
    <li><b>Run the App</b>:<br>
      <code>python app/app.py</code><br>
      <i>Set the environment variable if needed:</i><br>
      <code>set FLASK_SECRET_KEY=[your-secret-key]</code> (Windows)<br>
      <code>export FLASK_SECRET_KEY=[your-secret-key]</code> (Linux/Mac)
    </li>
    <li><b>Access</b>:<br>
      Open <a href="http://localhost:8080">http://localhost:8080</a>.
    </li>
  </ol>
</div>

## Deployment on a Cloud Platform (e.g., Railway)
<div>
  <ol>
    <li><b>Push to Docker Hub</b>:<br>
      <code>docker tag sms-spam-classifier:latest [your-username]/sms-spam-classifier:latest</code><br>
      <code>docker push [your-username]/sms-spam-classifier:latest</code><br>
      <i>Replace [your-username] with your Docker Hub username.</i>
    </li>
    <li><b>Set Up on Platform</b>:<br>
      Log into your cloud platform (e.g., Railway).<br>
      Create a new project and select "Empty project".<br>
      Add a service using the Docker image: <code>[your-username]/sms-spam-classifier:latest</code>.
    </li>
    <li><b>Configure</b>:<br>
      <ul>
        <li><b>Volume</b>: Add a volume (e.g., <code>spam-classifier-data</code>) mounted at <code>/data</code>.</li>
        <li><b>Variables</b>:<br>
          <code>FLASK_SECRET_KEY=[your-secret-key]</code><br>
          <code>DATA_DIR=/data</code>
        </li>
      </ul>
    </li>
    <li><b>Deploy</b>:<br>
      Deploy the service and access the provided URL (e.g., <code>[app-name].up.railway.app</code>).
    </li>
  </ol>
</div>

## Dependencies
<p>Defined in <code>requirements.txt</code>:</p>
<ul>
  <li>Flask==2.3.2</li>
  <li>Gunicorn==21.2.0</li>
  <li>Joblib==1.3.2</li>
  <li>Pandas==2.0.3</li>
  <li>Scikit-learn==1.3.0</li>
</ul>

## Notes
<ul>
  <li>Raw datasets in <code>data/raw</code> are for training; only <code>model.pkl</code> is included in the image.</li>
  <li>Runtime-generated files (e.g., <code>feedback.csv</code>) are stored in the <code>/data</code> volume.</li>
  <li>Retraining requires <code>src/data_processing/preprocess.py</code> and <code>src/model/train.py</code>.</li>
</ul>

## Contributing
<p>Fork this repository, submit pull requests, or report issues for feedback and improvements.</p>


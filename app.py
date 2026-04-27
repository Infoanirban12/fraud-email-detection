import os
import string
import joblib
import pandas as pd
import nltk
nltk.download('stopwords', quiet=True)


import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# ----------------------------
# Preprocess function
# ----------------------------
def preprocess(text):
    text = str(text).lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return ' '.join(words)

# ----------------------------
# Home Page
# ----------------------------
@app.route('/')
def home():
    return render_template('index.html')

# ----------------------------
# Train Model + Dashboard
# ----------------------------
@app.route('/train', methods=['POST'])
def train():
    file = request.files['file']

    if not file:
        return "No file uploaded!"

    df = pd.read_csv(file)

    # Fix column names
    if 'label' not in df.columns:
        df.columns = ['label', 'text']

    # Convert labels
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Preprocess
    df['clean_text'] = df['text'].apply(preprocess)

    # Vectorize
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Save model
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # ----------------------------
    # Save Confusion Matrix Graph
    # ----------------------------
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Spam', 'Spam'],
                yticklabels=['Not Spam', 'Spam'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs('static', exist_ok=True)
    plt.savefig(os.path.join('static', 'cm.png'))
    plt.close()

    return render_template('dashboard.html', accuracy=round(acc, 2))

# ----------------------------
# Predict Email
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['email']

    if not os.path.exists('model.pkl'):
        return "⚠️ Train model first!"

    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    text = preprocess(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)

    result = "🚨 Spam" if prediction[0] == 1 else "✅ Not Spam"

    return render_template('result.html', prediction=result)

# ----------------------------
# Run App
# ----------------------------

app.run(debug=True, use_reloader=False)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
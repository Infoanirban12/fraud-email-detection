# 📧 Fraud Email Detection System

## 🚀 Overview

This project is a **Machine Learning + Flask web application** that detects whether an email is **Spam (Fraud)** or **Not Spam (Ham)** using Natural Language Processing (NLP).

The system allows users to:

* Upload their own dataset (CSV)
* Train a machine learning model dynamically
* View model performance (accuracy + confusion matrix)
* Test email content in real-time

---

## 🧠 Tech Stack

* **Python**
* **Flask** (Backend)
* **Bootstrap** (Frontend UI)
* **Scikit-learn** (Machine Learning)
* **NLTK** (Text preprocessing)
* **Matplotlib & Seaborn** (Visualization)
* **Gunicorn** (Production server)

---

## 🤖 Machine Learning Pipeline

1. Text Preprocessing (lowercase, remove punctuation, stopwords)
2. Feature Extraction using **TF-IDF**
3. Classification using **Multinomial Naive Bayes**

---

## 📊 Features

* ✅ Upload CSV dataset
* ✅ Train model dynamically
* ✅ Accuracy display
* ✅ Confusion matrix visualization
* ✅ Live email prediction
* ✅ Clean Bootstrap UI

---

## 📂 Project Structure

```
fraud-email-flask/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── model.pkl
├── vectorizer.pkl
│
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── result.html
│
├── static/
│   └── cm.png
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/yourusername/fraud-email-detection.git
cd fraud-email-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the application

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

---

## 🌐 Deployment

This project is deployed using **Render**.



---

## 📈 Model Performance

* Accuracy: ~0.95 – 0.97
* Evaluated using confusion matrix, precision, and recall

---

## ⚠️ Limitations

* May not detect highly sophisticated phishing emails
* Performance depends on dataset quality

---

## 🚀 Future Improvements

* Add deep learning models (LSTM / BERT)
* Compare multiple ML models
* Improve UI/UX
* Add database support
* Real-time email integration

---

## 👨‍💻 Author

Anirban 
GitHub: https://github.com/Infoanirban12

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

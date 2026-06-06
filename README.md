# 📊 Ecommerce Review Sentiment Analysis & Issue Detection System.

AI-powered Sentiment Analysis & Issue Detection using Machine Learning


---

🚀 Overview

The Ecommerce Review Intelligence System is an end-to-end Machine Learning web application that analyzes customer reviews to extract actionable insights.

It performs:

Sentiment classification (Positive / Negative)

Issue detection using NLP-based keyword mapping

Recommendation generation for product improvement

Interactive analytics via a Streamlit dashboard


👉 Goal: Help businesses understand customer pain points at scale.


---

🧠 Key Capabilities

⚡ Real-time sentiment prediction

🔍 Automated issue detection from text

💡 Smart recommendation engine

📊 Interactive analytics dashboard

🌐 Fully deployed web application



---

🏗️ System Architecture

User Input Review
        ↓
Text Preprocessing (Cleaning, Normalization)
        ↓
TF-IDF Vectorization
        ↓
Machine Learning Model (LogReg / SVM / NB)
        ↓
Sentiment Prediction
        ↓
Rule-based Issue Detection
        ↓
Recommendation Engine
        ↓
Streamlit UI Output


---

📁 Project Structure

Ecommerce_Review_Project/
│
├── app.py                      # Streamlit application
├── sentiment_model.pkl        # Trained ML model
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
│
├── eda/                       # Exploratory Data Analysis
└── archive(1)/                # Raw dataset


---

🧠 Machine Learning Pipeline

1. Data Preprocessing

Lowercasing

Noise removal

Text cleaning using regex

Neutral review filtering


2. Feature Engineering

TF-IDF Vectorization (top 5000 features)


3. Model Training

Logistic Regression (Primary)

Support Vector Machine

Naive Bayes


4. Evaluation Metrics

Accuracy

Precision

Recall

F1-score



---

📊 Results

Logistic Regression achieved the best overall performance

TF-IDF effectively captured semantic importance in reviews

System successfully identifies key customer pain points:

🔋 Battery issues

📶 Connectivity problems

⚡ Performance lag

🛠️ Hardware failures




---

⚙️ How It Works

1. User enters a product review


2. Text is transformed using TF-IDF


3. ML model predicts sentiment


4. Rule-based engine detects issue category


5. Recommendation system suggests improvement


6. Output is displayed via Streamlit UI




---

🌐 Live Demo

👉 Try it here:
https://ecommerce-review-analyzer-gurd4tuxeqrcjv3ttmfzz8.streamlit.app/


---

🖥️ Tech Stack

Python 🐍

Scikit-learn 🤖

NLP (TF-IDF)

Pandas / NumPy

Streamlit 🌐

Matplotlib



---

📈 Business Impact

Helps companies analyze customer feedback at scale

Identifies product weaknesses early

Improves customer satisfaction through insights

Reduces manual review analysis effort



---

🚀 Future Improvements

Replace TF-IDF with BERT embeddings

Train deep learning classifier (LSTM/Transformers)

Add database for review storage

Add user authentication system

Deploy on AWS / GCP



---

👨‍💻 Author

Sphurthi Pudupakam
Aspiring Data Scientist | ML & NLP Enthusiast


---

⭐ If you like this project

Give the repository a ⭐ and feel free to fork it!


---

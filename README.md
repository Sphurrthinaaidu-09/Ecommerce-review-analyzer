# 📊 Ecommerce Review Intelligence System

An end-to-end Natural Language Processing (NLP) and Machine Learning application designed to transform unstructured customer reviews into actionable business intelligence. The system automatically identifies customer sentiment, detects product issues, and generates recommendations that help organizations improve customer satisfaction and product quality.

---
<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-NLP-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge">
</p><p align="center">
    <a href="https://ecommerce-review-analyzer-gurd4tuxeqrcjv3ttmfzz8.streamlit.app/">Live Demo</a> 
</p>

---

## 🚀 Project Overview

Modern e-commerce platforms receive thousands of customer reviews every day. While these reviews contain valuable insights, manually analyzing large volumes of feedback is inefficient and often impractical.

This project combines Natural Language Processing, Machine Learning, and Interactive Analytics to help organizations:
- Analyze customer sentiment at scale.
- Identify recurring product issues.
- Detect dissatisfaction drivers.
- Generate actionable product recommendations.
- Convert customer feedback into business intelligence.

The application functions as a complete **Customer Voice Analytics Platform** rather than a standalone sentiment classifier.

---

## 🎯 Business Problem

Organizations often struggle to understand what customers are saying across thousands of reviews. As review volume grows:
- Manual review becomes impossible.
- Critical issues go unnoticed.
- Product teams lose visibility into customer pain points.
- Customer satisfaction declines.

This system helps businesses move from reactive customer support to proactive product improvement by automatically extracting insights from review data.

---

## ⚙️ System Workflow

###  Step 1: Customer Review Ingestion
The platform supports two modes of analysis.

#### 👤 Individual Review Assessment
Users can enter a single product review into the application.
> *Example:* `"The battery drains very fast and charging is extremely slow."`
The system instantly analyzes the review and generates insights.

#### 📁 Bulk Review Analysis
Organizations can upload an entire CSV file containing thousands of customer reviews. The platform automatically processes all reviews and generates large-scale customer intelligence.

---

###  Step 2: NLP Processing Pipeline
Incoming reviews undergo multiple processing stages.
- **Text Cleaning:** Converts text to lowercase, removes special characters, and eliminates unnecessary noise.
- **Feature Engineering:** Customer reviews are converted into numerical representations using **TF-IDF (Term Frequency–Inverse Document Frequency)**, allowing machine learning models to understand textual patterns.

---

###  Step 3: Sentiment Prediction Engine
The processed reviews are passed through trained Machine Learning models (Logistic Regression, Support Vector Machine, or Naive Bayes) to accurately classify the sentiment baseline:
- 😊 Positive Review
- 😞 Negative Review

---

###  Step 4: Product Issue Detection
For negative reviews, the application performs issue categorization using NLP-based keyword mapping. Detected issue categories include:
-  Battery & Charging
-  Connectivity
-  Performance
-  Hardware Failure
-  Compatibility
-  Price & Value
-  Product Quality
-  User Interface & Controls

This helps organizations identify the root causes behind customer dissatisfaction.

---

###  Step 5: Recommendation Engine
The application automatically generates actionable recommendations based on detected issues:
- **Battery Issues:** *"Improve battery efficiency and charging reliability."*
- **Performance Issues:** *"Optimize software performance and reduce lag."*
- **Product Quality Issues:** *"Strengthen quality assurance and durability testing."*

This transforms raw customer complaints into practical improvement strategies.

---

###  Step 6: Executive Analytics Dashboard
For bulk review analysis, the platform generates business-friendly dashboards:
- **Sentiment Distribution:** Tracks positive vs. negative distributions alongside percentage breakdowns.
- **Issue Frequency Analysis:** Identifies the most common complaints, product weaknesses, and customer pain points.
- **Top Customer Concerns:** Ranks issue categories by occurrence frequency, enabling product managers to prioritize improvements.

---

###  Step 7: Exportable Intelligence Reports
After analysis, users can download fully processed reports containing:
- Review Text
- Predicted Sentiment
- Issue Category
- Generated Recommendation

---

## 📂 Dataset

**Dataset:** Amazon Electronics Product Reviews Dataset

### Dataset Characteristics:
- Thousands of customer reviews across the Electronics product category.
- Real-world, user-generated review text, ratings, and metadata.

### Key Features Evaluated:
- Age
- Department
- Job Role
- Monthly Income
- Job Satisfaction
- Environment Satisfaction
- Work-Life Balance
- Overtime
- Marital Status
- Total Working Years
- Years at Company

---

## 🔍 Exploratory Data Analysis (EDA)

Several customer behavior patterns were identified during analytics mapping:
- ✅ Battery-related complaints were among the most frequent negative review categories.
- ✅ Performance-related issues significantly influenced customer dissatisfaction.
- ✅ Connectivity problems frequently appeared in wireless devices.
- ✅ Positive reviews were primarily associated with usability and product experience.

### Visualizations Implemented:
- Employee Attrition Distribution
- Department vs Attrition
- Overtime vs Attrition
- Job Satisfaction vs Attrition
- Feature Importance Analysis

---

## ⚙️ Data Preprocessing

### Text Cleaning
- Lowercase conversion
- Special character removal
- Noise reduction

### Feature Engineering
- TF-IDF Vectorization

### Dataset Preparation
- Neutral reviews removed to protect classification boundaries.
- Balanced dataset preparation across Positive and Negative targets.

---

## 🤖 Machine Learning Models Evaluated

The following text classification models were trained and compared:
1. **Logistic Regression:** Implemented as a baseline NLP classifier.
2. **Naive Bayes:** Probabilistic text classification model optimized for word frequency structures.
3. **Support Vector Machine (SVM):** High-performance classification model for multi-dimensional textual data patterns.

---

## 📈 Model Evaluation

Models were rigorously compared using **Accuracy**, **Precision**, **Recall**, and **F1-Score**. The final production model was selected based on overall classification performance, stability, and minimized false deployment rates under text variances.

---

## 🔑 Business Insights Generated

The system helps organizations answer critical business questions:
- Why are customers unhappy?
- Which product issues occur most frequently?
- What improvements should product teams prioritize?
- How is customer sentiment changing?

This transforms raw customer reviews into strategic business intelligence.

---

## 🛠️ Technology Stack

- **Programming Language:** Python
- **NLP & Machine Learning:** Scikit-Learn, TF-IDF, Logistic Regression, Support Vector Machine, Naive Bayes
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Model Persistence:** Joblib
- **Deployment & Web App:** Streamlit

---

## 📸 Application Screenshots

<details>
<summary>🔍 1. Home Dashboard & Individual Review Analysis (Click to Expand)</summary>
<p align="center">
  <img src="https://github.com/user-attachments/assets/4a4db3d6-d75c-4f7e-987d-82eab7e7f50e" alt="Home Dashboard" width="85%" style="max-width: 100%; border-radius: 8px;" />
</p>
</details>

<details>
<summary>🔋 2. Issue Detection Results & Recommendations (Click to Expand)</summary>
<p align="center">
  <img src="https://github.com/user-attachments/assets/33fd88af-97c8-4bc4-bde3-a4aef38c2ef9" alt="Issue Detection" width="85%" style="max-width: 100%; border-radius: 8px;" />
</p>
</details>

<details>
<summary>📁 3. Bulk CSV Analysis & Upload Terminal (Click to Expand)</summary>
<p align="center">
  <img src="https://github.com/user-attachments/assets/61a3d780-8561-4984-91d6-4cf4a877179e" alt="Bulk Analytics File Upload" width="85%" style="max-width: 100%; border-radius: 8px;" />
</p>
</details>

<details>
<summary>📈 4. Executive Sentiment & Issue Frequency Dashboard (Click to Expand)</summary>
<p align="center">
  <img src="https://github.com/user-attachments/assets/ea0efd79-cca6-43a9-9a7c-f764ce938383" alt="Executive Dashboards" width="85%" style="max-width: 100%; border-radius: 8px;" />
</p>
</details>

---

## 📁 Project Structure

Ecommerce_Review_Project/
├── app.py                  # Core Streamlit application & interface logic
├── eda.py                  # Exploratory script configurations
├── sentiment_model.pkl     # Serialized Machine Learning Classifier
├── tfidf_vectorizer.pkl    # Trained text feature extractor matrix
├── requirements.txt        # Application package dependencies
└── README.md               # Documentation asset

## 🎓 Skills Demonstrated

Through this project, I gained and applied practical experience in:

- **Natural Language Processing (NLP):** Working with token-level tokenization pipelines, textual data cleanups, and string preprocessing techniques.
- **Text Classification:** Designing and configuring machine learning classification pipelines to systematically sort unstructured strings into discrete categorical labels.
- **Feature Engineering (TF-IDF):** Converting complex raw text corpuses into highly balanced, numerical spatial feature vectors using Term Frequency-Inverse Document Frequency.
- **Sentiment Analysis & Machine Learning:** Training, isolating, optimizing, and deploying multi-class algorithmic text classifiers.
- **Business Analytics & Customer Intelligence:** Bridging raw model sentiment probabilities with prescriptive actionable recommendations for product engineering teams.
- **Dashboard Development & Streamlit Deployment:** Serving interactive, production-ready analytics software applications to make machine learning models accessible to business stakeholders.

---

## 🔮 Future Enhancements

- **Transformer-Based Models (BERT):** Integrating deep learning context modeling to capture complex semantic patterns and contextual subtleties in customer reviews.
- **Aspect-Based Sentiment Analysis (ABSA):** Evaluating varying sub-sentiments across multiple product features or attributes mentioned inside a single review sentence.
- **Multi-Language Review Support:** Localizing preprocessing and modeling pipelines to analyze international customer feedback logs seamlessly.
- **Real-Time Customer Monitoring:** Directing automated data streaming pipelines to monitor live API market review ingestion streams continuously.
- **Enterprise Cloud Deployment:** Scaling local software runtimes into highly secure, containerized, and isolated cloud network environments.
- **Advanced Analytics Dashboard:** Introducing dynamic web charts and graph visualization panels for deeper business intelligence tracking over time.

---

## 👨‍💻 Author

**Sphurthhi Pudupakam** *Aspiring Data Analyst | Machine Learning Enthusiast* - **GitHub:** [https://github.com/Sphurrthinaaidu-09](https://github.com/Sphurrthinaaidu-09)  

---

⭐ If you found this project useful, consider giving the repository a star!

# 📊 Ecommerce Review Intelligence System

An end-to-end Natural Language Processing (NLP) and Machine Learning application designed to transform unstructured customer reviews into actionable business intelligence. The system automatically identifies customer sentiment, detects product issues, and generates recommendations that help organizations improve customer satisfaction and product quality.

---
<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-NLP-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge">
</p>
<p align="center">
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

### Step 1: Customer Review Ingestion
The platform supports two modes of analysis.

#### 👤 Individual Review Assessment
Users can enter a single product review into the application.
> *Example:* `"The battery drains very fast and charging is extremely slow."`
The system instantly analyzes the review and generates insights.

#### 📁 Bulk Review Analysis
Organizations can upload an entire CSV file containing thousands of customer reviews. The platform automatically processes all reviews and generates large-scale customer intelligence.

---

### Step 2: NLP Processing Pipeline
Incoming reviews undergo multiple processing stages.
- **Text Cleaning:** Converts text to lowercase, removes special characters, and eliminates unnecessary noise.
- **Feature Engineering:** Customer reviews are converted into numerical representations using **TF-IDF (Term Frequency–Inverse Document Frequency)**, allowing machine learning models to understand textual patterns.

---

### Step 3: Sentiment Prediction Engine
The processed reviews are passed through trained Machine Learning models (Logistic Regression, Support Vector Machine, or Naive Bayes) to accurately classify the sentiment baseline:
- 😊 Positive Review
- 😞 Negative Review

---

### Step 4: Product Issue Detection
For negative reviews, the application performs issue categorization using NLP-based keyword mapping. Detected issue categories include:
- Battery & Charging
- Connectivity
- Performance
- Hardware Failure
- Compatibility
- Price & Value
- Product Quality
- User Interface & Controls

This helps organizations identify the root causes behind customer dissatisfaction.

---

### Step 5: Recommendation Engine
The application automatically generates actionable recommendations based on detected issues:
- **Battery Issues:** *"Improve battery efficiency and charging reliability."*
- **Performance Issues:** *"Optimize software performance and reduce lag."*
- **Product Quality Issues:** *"Strengthen quality assurance and durability testing."*

This transforms raw customer complaints into practical improvement strategies.

---

### Step 6: Executive Analytics Dashboard
For bulk review analysis, the platform generates business-friendly dashboards:
- **Automated AI Core Insights:** Dynamically parses text configurations to pull the *Primary Operational Driver*, pinpoint *Friction Vectors*, and construct a *Dynamic Business Recommendation* without relying on hardcoded rules.
- **Sentiment Distribution:** Tracks positive vs. negative distributions alongside percentage breakdowns.
- **Issue Frequency Analysis:** Identifies the most common complaints, product weaknesses, and customer pain points using Pareto charting logic.

---

### Step 7: Exportable Intelligence Reports
After analysis, the integrated **Export Hub Pipeline** allows users to compile and download unique structural file layouts:
1. **Calculated Sentiment Ledger:** Best for database loading; contains granular individual review text columns alongside newly appended dynamic text feature analysis tags.
2. **Aggregated Executive Operational Logs:** Best for business stakeholders; generates a cross-tabulated performance metrics summary grouped cleanly by Product Line SKU.

---

## 📂 Dataset

**Dataset:** Amazon Electronics Product Reviews Dataset

### Target Inputs Evaluated:
- **ReviewText:** Raw textual feedback containing qualitative customer opinions.
- **Rating:** Numerical score metrics used to establish semantic ground truths.
- **Product:** Unique product identifier / SKU lines for relational grouping layouts.
- **Date:** Timestamps indicating when feedback text records were submitted.

---

## 🔍 Exploratory Data Analysis (EDA)

Several customer behavior patterns were identified during analytics mapping:
- Battery-related complaints were among the most frequent negative review categories.
- Performance-related issues significantly influenced customer dissatisfaction.
- Connectivity problems frequently appeared in wireless devices.
- Positive reviews were primarily associated with usability and product experience.

### Visualizations Implemented:
- Proportional Macro Sentiment Share (Donut / Pie Breakdowns)
- Pareto Distribution of Core Product Weaknesses (Categorical Bar Charts)
- Product-Line Crosstab Matrix Outlines

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
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Joblib
- **Deployment & Web App:** Streamlit

---

## 📸 Application Screenshots

<details>
<summary> 1. Home Gateway & Enterprise Data Ingestion Interface (Click to Expand)</summary>
<p align="center">
  <strong>CSV Dataset Ingestion & Target Variable Mapping:</strong><br>
  <img src="https://github.com/user-attachments/assets/c96310ac-d01d-427a-a77b-4306a2cd7bd2" width="100%" alt="Enterprise Data Ingestion Interface" />
  <br><em>Batch ingestion hub supporting multi-row CSV files where users can seamlessly parse document structural matrices and track processing footprints.</em>
</p>
</details>

<details>
<summary> 2. Dashboard Workspace & Executive Operations Terminal (Click to Expand)</summary>
<p align="center">
  <strong>High-Level Performance Metric Summaries & Operational Flags:</strong><br>
  <img src="https://github.com/user-attachments/assets/6b1acdd3-9104-4b85-b4b7-c462ec592639" width="100%" alt="Executive Operations Terminal" />
  <br><em>An executive monitoring space calculating operations volume, average score grades, customer health indexes, and automated AI core insights in runtime.</em>
</p>
</details>

<details>
<summary> 3. Structural Data View & Active Cache Master Registry (Click to Expand)</summary>
<p align="center">
  <strong>Ingested Data Streams Master Matrix Table:</strong><br>
  <img src="https://github.com/user-attachments/assets/989cc5bb-acd8-4831-b43c-888d91b2b397" width="100%" alt="Active Cache Master Registry" />
  <br><em>A direct validation sandbox allowing developers and stakeholders to cross-reference and skim raw parsed transaction records alongside review length evaluations.</em>
</p>
</details>

<details>
<summary> 4. Sentiment Playground (Click to Expand)</summary>
<p align="center">
  <strong>Live Single-Review Text Input Interface:</strong><br>
  <img src="https://github.com/user-attachments/assets/2e0d9615-49d2-48f8-a2c1-b22d230272f5" width="100%" alt="Real-Time Sentiment Playground" />
  <br><em>An ad-hoc interactive testing environment built to instantly execute classification pipeline steps for an individual custom customer text line.</em>
</p>
</details>

<details>
<summary> 5. Visual Deck Analytics Dashboard (Click to Expand)</summary>
<p align="center">
  <strong>Granular Segmentation Matrices & Visual Distribution Panels:</strong><br>
  <img src="https://github.com/user-attachments/assets/b03ec974-c73f-49af-a1f5-6eedf6ce76d3" width="100%" alt="Visual Analytics Deck" />
  <br><em>Plots macro sentiment distribution shares using proportional charts alongside item-by-item volume tracking arrays.</em>
</p>
</details>

<details>
<summary> 6. Review Matrix Grid Explorer (Click to Expand)</summary>
<p align="center">
  <strong>Discovered Operational Registry Indices & Core Filtering:</strong><br>
  <img src="https://github.com/user-attachments/assets/241186dd-815c-414b-97ae-64c0bf90c934" width="100%" alt="Review Matrix Grid Explorer" />
  <br><em>Granular data discovery deck utilizing keyword substring search maps and categorical filters to rapidly review parsed system targets.</em>
</p>
</details>

<details>
<summary> 7. Downstream Report Export Pipeline Center (Click to Expand)</summary>
<p align="center">
  <strong>Integrated Session Compilation Export Terminal:</strong><br>
  <img src="https://github.com/user-attachments/assets/e7592110-33d8-4f57-87af-eb62852f8b4d" width="100%" alt="Downstream Report Export Pipeline Center" />
  <br><em>Session transform compiling station enabling structured file outputs as Granular Sentiment Ledgers or Aggregated Executive Logs.</em>
</p>
</details>

---

## 📁 Project Structure

Ecommerce_Review_Project/
│
├── app.py                  # Main Streamlit Application UI & Pipelines
├── eda.py                  # Analytical exploration scripts & charting
├── sentiment_model.pkl     # Persisted baseline ML classifier weights
├── tfidf_vectorizer.pkl    # Serialized Text Vectorization configurations
├── requirements.txt        # Managed software package dependencies
└── README.md               # Portfolio presentation file

---

## 🎓 Skills Demonstrated

Through this project, I gained and applied practical experience in:
Natural Language Processing (NLP): Working with token-level tokenization pipelines, textual data cleanups, and string preprocessing techniques.
Text Classification: Designing and configuring machine learning classification pipelines to systematically sort unstructured strings into discrete categorical labels.
Feature Engineering (TF-IDF): Converting complex raw text corpuses into highly balanced, numerical spatial feature vectors using Term Frequency-Inverse Document Frequency.
Sentiment Analysis & Machine Learning: Training, isolating, optimizing, and deploying multi-class algorithmic text classifiers.
Business Analytics & Customer Intelligence: Bridging raw model sentiment probabilities with prescriptive actionable recommendations for product engineering teams.
Dashboard Development & Streamlit Deployment: Serving interactive, production-ready analytics software applications to make machine learning models accessible to business stakeholders.

---

## 🔮 Future Enhancements
Transformer-Based Models (BERT): Integrating deep learning context modeling to capture complex semantic patterns and contextual subtleties in customer reviews.
Aspect-Based Sentiment Analysis (ABSA): Evaluating varying sub-sentiments across multiple product features or attributes mentioned inside a single review sentence.
Multi-Language Review Support: Localizing preprocessing and modeling pipelines to analyze international customer feedback logs seamlessly.
Real-Time Customer Monitoring: Directing automated data streaming pipelines to monitor live API market review ingestion streams continuously.
Enterprise Cloud Deployment: Scaling local software runtimes into highly secure, containerized, and isolated cloud network environments.
Advanced Analytics Dashboard: Introducing dynamic web charts and graph visualization panels for deeper business intelligence tracking over time.

## 👨‍💻 Author
Sphurrthi Naidu Aspiring Data Analyst | Machine Learning Enthusiast 
GitHub: [https://github.com/Sphurrthinaaidu-09]

⭐ If you found this project useful, consider giving the repository a star!


import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Issue Detection
issue_keywords = {
    "Battery_Charging": [
        "battery", "charge", "charging",
        "drain", "drains", "drained",
        "power", "backup", "battery life",
        "dies", "dead battery"
    ],

    "Connectivity": [
        "connection", "connectivity",
        "disconnect", "disconnecting",
        "network", "internet",
        "wifi", "wi-fi",
        "bluetooth", "signal"
    ],

    "Performance": [
        "slow", "lag", "lags",
        "lagging", "freeze",
        "freezes", "frozen",
        "crash", "crashes",
        "hanging", "hang",
        "performance"
    ],

    "Hardware_Failure": [
        "broken", "defective",
        "failure", "failed",
        "stopped working",
        "dead", "damaged",
        "malfunction", "faulty"
    ],

    "User_Interface": [
        "button", "buttons",
        "sticky", "stuck",
        "click", "clicking",
        "responsive", "unresponsive",
        "respond", "response",
        "touch", "input",
        "control", "controls"
    ],

    "Display_Issue": [
        "display", "screen",
        "pixel", "pixels",
        "brightness", "flicker",
        "flickering", "black screen"
    ],

    "Audio_Issue": [
        "speaker", "speakers",
        "sound", "audio",
        "volume", "microphone",
        "mic", "noise"
    ],

    "Compatibility": [
        "compatible", "compatibility",
        "support", "windows",
        "android", "ios",
        "device", "platform"
    ],

    "Price_Value": [
        "price", "pricing",
        "expensive", "costly",
        "money", "worth",
        "overpriced", "value"
    ],

    "Product_Quality": [
        "poor quality",
        "quality", "terrible",
        "horrible", "junk",
        "cheap", "flimsy",
        "disappointed"
    ]
}

# Recommendations
recommendations = {
    "Battery_Charging":
        "Improve battery life and charging reliability.",

    "Connectivity":
        "Improve network stability, Wi-Fi, and Bluetooth connectivity.",

    "Performance":
        "Optimize software performance and reduce lag, crashes, and freezes.",

    "Hardware_Failure":
        "Improve hardware durability and quality testing.",

    "User_Interface":
        "Improve button responsiveness, controls, and user interaction experience.",

    "Display_Issue":
        "Improve display quality, brightness stability, and screen reliability.",

    "Audio_Issue":
        "Improve speaker output, audio quality, and microphone performance.",

    "Compatibility":
        "Improve compatibility across devices and operating systems.",

    "Price_Value":
        "Review pricing strategy and improve value for money.",

    "Product_Quality":
        "Improve overall product quality and customer experience.",

    "Other":
        "Further investigation required."
}

# Detect issue
def detect_issue(review):
    review = review.lower()

    for issue, keywords in issue_keywords.items():
        for word in keywords:
            if word in review:
                return issue

    return "Other"


# Page title
st.title("🛒 E-Commerce Review Analyzer")

review = st.text_area("Enter Product Review")

if st.button("Analyze"):

    review_vec = tfidf.transform([review])

    sentiment = model.predict(review_vec)[0]

    st.subheader("Prediction Result")

    st.write("### Sentiment:")
    st.success(sentiment)

    if sentiment == "Negative":

        issue = detect_issue(review)

        st.write("### Issue Category:")
        st.error(issue)

        st.write("### Recommendation:")
        st.info(recommendations.get(issue))

    else:

        st.success("Customer is satisfied 😊")


uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

import pandas as pd

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    review_column = st.selectbox(
        "Select Review Text Column",
        data.columns
    )

    if st.button("Analyze CSV"):

        # Sentiment Prediction
        review_vectors = tfidf.transform(
            data[review_column].astype(str)
        )

        data["Sentiment"] = model.predict(review_vectors)

        # Issue Detection
        data["Issue"] = data.apply(
            lambda x:
            detect_issue(x[review_column])
            if x["Sentiment"] == "Negative"
            else "No Issue",
            axis=1
        )

        # Recommendation
        data["Recommendation"] = data["Issue"].apply(
            lambda issue:
            recommendations.get(
                issue,
                "Customer is satisfied 😊"
            )
        )

        st.success("Analysis Complete ✅")

        st.dataframe(data.head(20))

        csv_data = data.to_csv(index=False)

        # Save locally
        data.to_csv("review_analysis_results.csv", index=False)

        st.success("Results saved as review_analysis_results.csv")

        # Download button
        st.download_button(
            label="📥 Download Results",
            data=csv_data,
            file_name="review_analysis_results.csv",
            mime="text/csv"
        )

        # ==========================
        # DASHBOARD INSIGHTS
        # ==========================

        st.subheader("📊 Dashboard Insights")

        total_reviews = len(data)

        positive_reviews = len(
            data[data["Sentiment"] == "Positive"]
        )

        negative_reviews = len(
            data[data["Sentiment"] == "Negative"]
        )

        # Percentages
        positive_percent = round(
            (positive_reviews / total_reviews) * 100,
            2
        )

        negative_percent = round(
            (negative_reviews / total_reviews) * 100,
            2
        )

        # KPI Cards
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Reviews", total_reviews)
        col2.metric("Positive Reviews", positive_reviews)
        col3.metric("Negative Reviews", negative_reviews)

        # Percentage Cards
        col4, col5 = st.columns(2)

        col4.metric(
            "Positive %",
            f"{positive_percent}%"
        )

        col5.metric(
            "Negative %",
            f"{negative_percent}%"
        )

        # Overall Sentiment
        if positive_reviews > negative_reviews:
            st.success(
                "Overall Customer Sentiment is Positive 😊"
            )
        else:
            st.error(
                "Overall Customer Sentiment is Negative 😟"
            )

        # Issue Analysis
        negative_data = data[
            data["Sentiment"] == "Negative"
        ]

        if len(negative_data) > 0:

            issue_counts = negative_data["Issue"].value_counts()

            top_issue = issue_counts.idxmax()

            st.warning(
                f"🚨 Top Customer Issue: {top_issue}"
            )

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()

            issue_counts.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Issue Frequency")
            ax.set_xlabel("Issue Type")
            ax.set_ylabel("Count")

            st.pyplot(fig)

        # Pie Chart
        st.subheader("🥧 Sentiment Distribution")

        sentiment_counts = data["Sentiment"].value_counts()

        fig2, ax2 = plt.subplots()

        ax2.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%"
        )

        ax2.set_title(
            "Positive vs Negative Reviews"
        )

        st.pyplot(fig2)
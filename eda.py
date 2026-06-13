import re

import pandas as pd

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_json("archive (1)/Electronics_5.json", lines=True)
print(df.head())
print("\nColumns:")
print(df.columns)

print(df[['reviewText','overall']].head(10))

def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating <= 2:
        return "Negative"
    else:
        return "Neutral"
df["sentiment"] = df["overall"].apply(get_sentiment)
print(df[["overall", "sentiment"]].head(10))

df = df[df["sentiment"] != "Neutral"]
print(df["sentiment"].value_counts())

df = df[df["sentiment"] != "Neutral"]
df_sample = df.sample(n=50000, random_state=42)
print(df_sample.shape)

print(df_sample["sentiment"].value_counts())

print(df_sample[["reviewText", "sentiment"]].head())

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
df_sample["clean_review"] = df_sample["reviewText"].apply(clean_text)
print(df_sample[["reviewText", "clean_review"]].head())

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_sample["clean_review"])
y = df_sample["sentiment"]
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Balanced Logistic Regression")
print(classification_report(y_test, y_pred))

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

print("Naive Bayes")
print(classification_report(y_test, y_pred_nb))

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("SVM")
print(classification_report(y_test, y_pred_svm))

negative_reviews = df_sample[df_sample["sentiment"] == "Negative"]
print("Negative Reviews:", negative_reviews.shape)
print(negative_reviews["reviewText"].head())

for review in negative_reviews["reviewText"].head(5):
    print("\n")
    print(review)
    print("-"*100)

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

def detect_issue(review):
    review = str(review).lower()
    for issue, keywords in issue_keywords.items():
        for keyword in keywords:
            if keyword in review:
                return issue
    return "Other"
negative_reviews["Issue"] = negative_reviews["reviewText"].apply(detect_issue)
print(
    negative_reviews[
        ["reviewText", "Issue"]
    ].head(10)
)
print(negative_reviews["Issue"].value_counts())


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


negative_reviews["Recommendation"] = (
    negative_reviews["Issue"]
    .map(recommendations)
)
print(
    negative_reviews[
        ["Issue", "Recommendation"]
    ].head(10)
)

def predict_review(review):

    review_vec = tfidf.transform([review])
    sentiment = model.predict(review_vec)[0]

    issue = detect_issue(review)

    print("\nSentiment:", sentiment)

    if sentiment == "Negative":

        print("Issue:", issue)
        print("Recommendation:",
              recommendations.get(issue,
              "No recommendation available"))

    elif issue != "Other":

        print("Possible Issue Detected:", issue)
        print("Recommendation:",
              recommendations.get(issue,
              "No recommendation available"))

    else:

        print("No issues detected. Customer is satisfied 😊")

print(type(model))

predict_review("This product works perfectly and I love it")
predict_review("Battery drains very fast and charging is slow")

predict_review("This product is horrible and a waste of money")
predict_review("Terrible quality. Very disappointed.")
predict_review("The device is defective and completely broken")

import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model Saved Successfully!")
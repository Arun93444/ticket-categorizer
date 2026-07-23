import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Download stopwords (first run only)
nltk.download("stopwords")

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/support_tickets.csv")

print("=" * 60)
print("Dataset Loaded Successfully")
print(df.head())
print("=" * 60)

# -----------------------------
# Combine Subject + Body
# -----------------------------
df["text"] = df["subject"] + " " + df["body"]

# -----------------------------
# Stopwords
# -----------------------------
stop_words = set(stopwords.words("english"))

# -----------------------------
# Cleaning Function
# -----------------------------
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\d+", "", text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# Apply Cleaning
df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------
# Features & Labels
# -----------------------------
X = df["clean_text"]

y = df["category"]

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(f"Training Samples : {X_train.shape[0]}")
print(f"Testing Samples  : {X_test.shape[0]}")

# ===================================================
# Model 1 : Naive Bayes
# ===================================================

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_acc = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Accuracy :", round(nb_acc * 100, 2), "%")

# ===================================================
# Model 2 : Logistic Regression
# ===================================================

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy :", round(lr_acc * 100, 2), "%")

# ===================================================
# Select Best Model
# ===================================================

if lr_acc > nb_acc:

    best_model = lr_model
    best_name = "Logistic Regression"
    best_pred = lr_pred
    best_acc = lr_acc

else:

    best_model = nb_model
    best_name = "Multinomial Naive Bayes"
    best_pred = nb_pred
    best_acc = nb_acc

print("\n" + "=" * 60)
print("Best Model :", best_name)
print("Accuracy :", round(best_acc * 100, 2), "%")
print("=" * 60)

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report\n")

print(classification_report(y_test, best_pred))

# -----------------------------
# Confusion Matrix
# -----------------------------
print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, best_pred))

# -----------------------------
# Save Best Model
# -----------------------------
joblib.dump(best_model, "models/model.pkl")

joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully!")

print("models/model.pkl")

print("models/vectorizer.pkl")
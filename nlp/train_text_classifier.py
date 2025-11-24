"""
Train/test a text classifier on feedback for Intelligent Business AI Analyst.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Load dataset
df = pd.read_csv("data/business_data.csv")

# Features and target
X = df['feedback']          # text column
y = df['target_success']    # target binary

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression
clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_train_vec, y_train)

# Evaluate
y_pred = clf.predict(X_test_vec)
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Save model and vectorizer
os.makedirs('nlp/models', exist_ok=True)
joblib.dump(clf, 'nlp/models/text_classifier.pkl')
joblib.dump(vectorizer, 'nlp/models/tfidf_vectorizer.pkl')
print("Text classifier and vectorizer saved to nlp/models/")


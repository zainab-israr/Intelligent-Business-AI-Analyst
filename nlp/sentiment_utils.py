"""
Sentiment analysis module for Intelligent Business AI Analyst.

Uses TF-IDF + Logistic Regression classifier trained on feedback.
CPU-friendly, Mac-safe, no transformers required.
"""

import pandas as pd
import joblib
import os

# Load trained text classifier and TF-IDF vectorizer
clf_path = "nlp/models/text_classifier.pkl"
vec_path = "nlp/models/tfidf_vectorizer.pkl"

if not os.path.exists(clf_path) or not os.path.exists(vec_path):
    raise FileNotFoundError("Please run nlp/train_text_classifier.py first to generate model and vectorizer.")

clf = joblib.load(clf_path)
vectorizer = joblib.load(vec_path)

def predict_sentiment(text_list):
    """
    Predict sentiment for a list of texts using trained classifier.
    
    Args:
        text_list (list of str)
    
    Returns:
        List of dicts: [{'text': ..., 'label': ..., 'score': ...}, ...]
    """
    results = []
    X_vec = vectorizer.transform(text_list)
    probs = clf.predict_proba(X_vec)
    preds = clf.predict(X_vec)
    for i, text in enumerate(text_list):
        results.append({
            "text": text,
            "label": "POSITIVE" if preds[i]==1 else "NEGATIVE",
            "score": probs[i][preds[i]]
        })
    return results

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/business_data.csv")
    feedbacks = df['feedback'].tolist()[:5]
    sentiments = predict_sentiment(feedbacks)
    for s in sentiments:
        print(f"Text: {s['text']}")
        print(f"Sentiment: {s['label']} (score: {s['score']:.2f})\n")

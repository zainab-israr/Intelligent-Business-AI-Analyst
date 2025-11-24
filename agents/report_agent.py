"""
Intelligent Business AI Analyst Agent
Combines ML predictions, NLP sentiment, and RAG document retrieval
to generate a business insights report.
"""

import pandas as pd
import joblib
from nlp.sentiment_utils import predict_sentiment
from rag.query_rag import query_rag

# Load ML model and preprocessor
ml_model = joblib.load("ml/model.pkl")
preprocessor = joblib.load("ml/preprocessor.pkl")

# Load dataset
df = pd.read_csv("data/business_data.csv")

def generate_report(df, top_docs=2):
    report = []

    # Preprocess features
    X = df.drop(columns=["customer_id", "feedback", "target_success"])
    X_processed = preprocessor.transform(X)

    # ML predictions
    preds = ml_model.predict(X_processed)
    df['predicted_success'] = preds

    # NLP sentiment
    sentiments = predict_sentiment(df['feedback'].tolist())
    df['feedback_sentiment'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    # Build report
    for idx, row in df.iterrows():
        entry = {
            "customer_id": row['customer_id'],
            "predicted_success": row['predicted_success'],
            "feedback": row['feedback'],
            "sentiment": row['feedback_sentiment'],
            "sentiment_score": row['sentiment_score']
        }

        # Query RAG for relevant docs
        docs = query_rag(row['feedback'], top_k=top_docs)
        entry['relevant_docs'] = [d['doc_name'] for d in docs]

        report.append(entry)

    report_df = pd.DataFrame(report)
    report_file = "business_insights_report.csv"
    report_df.to_csv(report_file, index=False)
    print(f"Business Insights Report saved: {report_file}")
    return report_df

if __name__ == "__main__":
    report_df = generate_report(df)
    print(report_df.head())

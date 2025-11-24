import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml.feature_engineering import preprocessor
from nlp.sentiment_utils import predict_sentiment
from rag.query_rag import query_rag



import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
print("Project root added to sys.path:", PROJECT_ROOT)


# Set page config
st.set_page_config(page_title="Intelligent Business AI Analyst", layout="wide")

# Load ML model
ml_model = joblib.load("ml/model.pkl")

st.title("Intelligent Business AI Analyst")
st.markdown("""
This app analyzes your business data:
- Predicts customer/transaction success
- Performs sentiment analysis on feedback
- Retrieves relevant business documents
- Generates a downloadable insights report
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload business CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ML Predictions
    X = df.drop(columns=["customer_id", "feedback", "target_success"], errors='ignore')
    X_processed = preprocessor.transform(X)
    preds = ml_model.predict(X_processed)
    df['predicted_success'] = preds

    st.subheader("ML Predictions")
    def highlight_success(val):
        color = 'lightgreen' if val==1 else 'lightcoral'
        return f'background-color: {color}'
    st.dataframe(df[['customer_id', 'predicted_success']].style.applymap(highlight_success))

    # Feature Importance Chart
    st.subheader("Feature Importance")
    model = ml_model
    preproc = preprocessor
    # numeric + one-hot features
    numeric_features = ['sales_amount', 'units_sold', 'marketing_spend',
                        'customer_age', 'loyalty_score', 'visits_last_month',
                        'avg_purchase_value', 'discounts_used']
    cat_features = preproc.transformers_[1][1]['onehot'].get_feature_names_out(['region','product_category'])
    feature_names = numeric_features + list(cat_features)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices], ax=ax)
    ax.set_title("Feature Importances")
    st.pyplot(fig)

    # NLP Sentiment Analysis
    st.subheader("Feedback Sentiment")
    sentiments = predict_sentiment(df['feedback'].tolist())
    df['feedback_sentiment'] = [s['label'] for s in sentiments]
    df['sentiment_score'] = [s['score'] for s in sentiments]

    def highlight_sentiment(val):
        color = 'lightgreen' if val=='POSITIVE' else 'lightcoral'
        return f'background-color: {color}'

    st.dataframe(df[['customer_id', 'feedback', 'feedback_sentiment', 'sentiment_score']].style.applymap(highlight_sentiment, subset=['feedback_sentiment']))

    # Sentiment distribution chart
    st.subheader("Sentiment Distribution")
    fig2, ax2 = plt.subplots()
    df['feedback_sentiment'].value_counts().plot(kind='bar', color=['lightcoral', 'lightgreen'], ax=ax2)
    ax2.set_ylabel("Count")
    st.pyplot(fig2)

    # RAG Document Retrieval
    st.subheader("RAG Document Retrieval")
    question = st.text_input("Ask a question about your business data:")
    if question:
        results = query_rag(question)
        st.write("Top relevant documents:")
        for r in results:
            st.write(f"- {r['doc_name']} (distance: {r['distance']:.4f})")

    # Downloadable report
    st.subheader("📥 Download Business Insights Report")
    report_file = "business_insights_report.csv"
    df.to_csv(report_file, index=False)
    st.download_button("Download Report CSV", report_file)

# Intelligent Business AI Analyst

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/<your-username>/Intelligent-Business-AI-Analyst/main/app/streamlit_app.py)

A comprehensive AI-driven project that analyzes business data using **Machine Learning (ML)**, **Natural Language Processing (NLP)**, **Generative AI**, and **Retrieval-Augmented Generation (RAG)**. It provides interactive business insights, sentiment analysis, and intelligent report generation via a **Streamlit web interface**.

---


## **Features**

- **Business Data Analysis:** Predict trends, outcomes, and key metrics using ML models.
- **Sentiment Analysis:** Classify feedback or text into positive, negative, or neutral categories using NLP.
- **Document Retrieval (RAG):** Retrieve top relevant business documents (e.g., reports, policies) using vector embeddings.
- **Agentic AI:** Combine predictions, document retrieval, and sentiment analysis for end-to-end business insights.
- **Interactive Streamlit UI:** Visualize predictions, tables, charts, and download generated reports.
- **Generative AI Components:** Automatically summarize retrieved business documents and generate insights.

---

## **Demo**

### GIF Demo (Streamlit App)

![Streamlit Demo](streamlit_demo.gif)  

---

## **Project Structure**

```
Intelligent-Business-AI-Analyst/
│
├─ app/ # Streamlit application
│ └─ streamlit_app.py
│
├─ data/ # Sample or synthetic data
│ └─ business_data.csv
│
├─ docs/ # RAG documents for retrieval
│ ├─ monthly_report.txt
│ └─ hr_policy.txt
│
├─ ml/ # ML model training and preprocessing
│ ├─ feature_engineering.py
│ ├─ training.py
│ ├─ model.pkl
│ ├─ preprocessor.pkl
│ └─ init.py
│
├─ nlp/ # Sentiment analysis
│ ├─ sentiment_utils.py
│ ├─ models/
│ │ ├─ text_classifier.pkl
│ │ └─ tfidf_vectorizer.pkl
│ └─ init.py
│
├─ rag/ # RAG vector store and query
│ ├─ build_vectorstore.py
│ ├─ query_rag.py
│ ├─ faiss_index.bin
│ └─ doc_names.pkl
│
├─ agents/ # End-to-end agent orchestration
│ ├─ report_agent.py
│ └─ init.py
│
├─ .gitignore
├─ README.md
└─ requirements.txt
```
---

## **Future Improvements**

- Add more generative AI summaries for reports

- Improve UI with dynamic charts and filtering

- Deploy on cloud platforms with authentication

- Add more diverse datasets and document types for RAG

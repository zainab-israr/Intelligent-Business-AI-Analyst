# Intelligent Business AI Analyst

Complete ready-to-upload repository demonstrating ML, Data Science, NLP, GenAI (RAG) and a simple Agent.

Quickstart:
1. python -m venv .venv && source .venv/bin/activate
2. pip install -r requirements.txt
3. python data/generate_synthetic_data.py
4. python ml/training.py
5. python rag/build_vectorstore.py --docs docs/
6. streamlit run app/streamlit_app.py
7. uvicorn api.fastapi_app:app --reload --port 8000


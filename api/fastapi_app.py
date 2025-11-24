from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
from nlp.sentiment_utils import load_sentiment_model, predict_sentiment
from rag.query_rag import RAG

app = FastAPI()
model = None
sent_tok, sent_model = None, None
rag = None

@app.on_event('startup')
def load_components():
    global model, sent_tok, sent_model, rag
    try:
        model = joblib.load('ml/model.pkl')
    except Exception as e:
        print('Model load error', e)
    try:
        sent_tok, sent_model = load_sentiment_model()
    except Exception as e:
        print('Sentiment model load error', e)
    try:
        rag = RAG()
    except Exception as e:
        print('RAG load error', e)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    X = df[[c for c in df.columns if c not in ['employee_id','target_churn','feedback']]]
    preds = model.predict_proba(X)[:,1].tolist() if model is not None else []
    return {'predictions': preds}

@app.post('/chat')
async def chat(query: dict):
    q = query.get('q')
    docs = rag.query(q) if rag is not None else []
    return {'answer_docs': docs}

@app.get('/sentiment')
def sentiment(q: str):
    labels, probs = predict_sentiment([q], sent_tok, sent_model)
    return {'label': int(labels[0]), 'probs': probs.tolist()}

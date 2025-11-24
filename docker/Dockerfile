FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip && pip install -r requirements.txt
EXPOSE 8501 8000
CMD ["bash","-c","streamlit run app/streamlit_app.py --server.port 8501 & uvicorn api.fastapi_app:app --host 0.0.0.0 --port 8000"]

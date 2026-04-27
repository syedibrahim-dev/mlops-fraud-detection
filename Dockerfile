FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/model.json ./data/model.json
COPY data/feature_names.pkl ./data/feature_names.pkl

ENV MODEL_PATH=data/model.json
ENV MODEL_VERSION=1

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]

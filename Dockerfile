FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
EXPOSE 8000
# The first start trains a small synthetic demo model if models/ml/ml_bundle.joblib is missing.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

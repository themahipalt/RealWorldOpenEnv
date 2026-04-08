FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY customer_support_env/ ./customer_support_env/
COPY baseline.py .
COPY inference.py .
COPY app.py .
COPY openenv.yaml .

EXPOSE 7860

CMD ["python", "app.py"]

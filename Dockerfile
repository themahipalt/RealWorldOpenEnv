FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY customer_support_env/ ./customer_support_env/
COPY baseline.py .
COPY inference.py .
COPY openenv.yaml .

# Default: run inference script
CMD ["python", "inference.py"]

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY customer_support_env/ ./customer_support_env/
COPY baseline.py .
COPY openenv.yaml .

# Default: run baseline against all tasks
# Override with: docker run <image> python -c "..."
CMD ["python", "baseline.py", "--episodes", "3"]

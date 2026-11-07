FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://pypi.mydigipay.info/simple

COPY ingestion.py schema.sql .

CMD ["python", "ingestion.py"]

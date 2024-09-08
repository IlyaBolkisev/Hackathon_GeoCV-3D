# app/Dockerfile
FROM python:3.11-slim

WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY geo_estimate geo_estimate

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

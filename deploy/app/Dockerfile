# app/Dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


WORKDIR /src

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY modules modules

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

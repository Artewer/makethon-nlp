# base python image
FROM python:3.10-slim as base

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#TODO what is "rm -rf /var/lib/apt/lists/*" for? 
RUN apt-get update

WORKDIR /src

# Install dependencies
COPY requirements.txt .
RUN apt-get update -y && apt-get install -y gcc && \
    pip install -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

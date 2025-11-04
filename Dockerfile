FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        python3-dev \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY telco_customer_churn_analysis ./telco_customer_churn_analysis

ENV PYTHONPATH=/app/telco_customer_churn_analysis/src

EXPOSE 5000

CMD ["python", "telco_customer_churn_analysis/src/application.py"]
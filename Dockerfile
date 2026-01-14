FROM python:3.8-slim 

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY results/ ./results/

ENV PYTHONPATH=/app
ENV AAE_PATH=/app/results/ae_model.pkl

CMD ["python", "src/run_experiment.py", "--dataset", "data/raw/dataset.pkl", "--ae-path", "results/ae_model.pkl"]

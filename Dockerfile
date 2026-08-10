FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY scripts/download_model.py ./scripts/download_model.py
RUN python scripts/download_model.py
COPY src ./src
COPY models ./models
ENTRYPOINT ["python", "-u", "-m", "src.main"]

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV HF_HOME=/data/.cache/huggingface
ENV TORCH_HOME=/data/.cache/torch

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && \
    pip install -r requirements.txt

COPY main.py plant_gate.py ./
COPY model/best_model.pth model/best_model.pth
COPY model/best_model_v5.pth model/best_model_v5.pth

EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]